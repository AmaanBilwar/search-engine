import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from google import genai
from dotenv import load_dotenv

load_dotenv()

IO_THREAD_POOL_SIZE = max(10, min(32, (os.cpu_count() or 4) * 4))


class AsyncVideoTranscriber:
    """Async video transcriber using direct video upload."""
    
    _executor: Optional[ThreadPoolExecutor] = None
    _executor_refs = 0
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash",
        max_workers: Optional[int] = None
    ):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.model = model
        self._client: Optional[genai.Client] = None
        self._uploaded_files: Dict[str, Any] = {}
        self._file_state_cache: Dict[str, tuple[Any, float]] = {}
        self._max_workers = max_workers
        self._own_executor = max_workers is not None
        self._executor: Optional[ThreadPoolExecutor] = None
    
    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client
    
    @classmethod
    def _get_executor(cls, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get or create ThreadPoolExecutor."""
        if max_workers is not None:
            return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VideoTranscriber")
        
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=IO_THREAD_POOL_SIZE,
                thread_name_prefix="VideoTranscriber-Shared"
            )
        cls._executor_refs += 1
        return cls._executor
    
    @classmethod
    def _release_executor(cls) -> None:
        """Release reference to shared executor."""
        cls._executor_refs -= 1
        if cls._executor_refs <= 0 and cls._executor is not None:
            cls._executor.shutdown(wait=False)
            cls._executor = None
            cls._executor_refs = 0
    
    def _get_my_executor(self) -> ThreadPoolExecutor:
        """Get executor for this instance."""
        if self._own_executor:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="VideoTranscriber-Instance"
                )
            return self._executor
        else:
            return self._get_executor(None)
    
    async def _run_in_executor(self, func) -> Any:
        """Run blocking operations in thread pool executor."""
        executor = self._get_my_executor()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func)
    
    async def _verify_file_exists(self, file_path: str) -> None:
        """Verify file exists."""
        path = Path(file_path)
        if not await asyncio.to_thread(path.exists):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        if not await asyncio.to_thread(path.is_file):
            raise ValueError(f"Path is not a file: {file_path}")
    
    async def _get_file_size(self, file_path: str) -> int:
        """Get file size."""
        return await asyncio.to_thread(os.path.getsize, file_path)
    
    async def _get_file_state(self, file_name: str, use_cache: bool = True) -> Any:
        """Get file state from Gemini API."""
        if use_cache and file_name in self._file_state_cache:
            state, timestamp = self._file_state_cache[file_name]
            if time.time() - timestamp < 5.0:
                return state
        
        state = await self._run_in_executor(
            partial(self.client.files.get, name=file_name)
        )
        
        if use_cache:
            self._file_state_cache[file_name] = (state, time.time())
        
        return state
    
    async def _wait_for_file_active(
        self,
        file_name: str,
        poll_interval: float = 0.5,
        max_wait_time: float = 300.0,
        initial_delay: float = 0.0
    ) -> Any:
        """Poll until uploaded file is in ACTIVE state."""
        start_time = time.time()
        poll_count = 0
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_wait_time:
                raise TimeoutError(
                    f"File {file_name} did not become ACTIVE within {max_wait_time}s. "
                    f"Last checked after {poll_count} polls."
                )
            
            try:
                file_info = await self._get_file_state(file_name, use_cache=(poll_count > 0))
                
                if hasattr(file_info, 'state'):
                    state_obj = file_info.state
                    if hasattr(state_obj, 'name'):
                        state = state_obj.name
                    elif hasattr(state_obj, 'value'):
                        state = state_obj.value
                    else:
                        state = str(state_obj)
                elif hasattr(file_info, 'state_name'):
                    state = file_info.state_name
                else:
                    state = getattr(file_info, 'state', 'UNKNOWN')
                    if not isinstance(state, str):
                        state = str(state)
                
                poll_count += 1
                
                if state == 'ACTIVE':
                    if poll_count > 1:
                        print(f"File {file_name} is now ACTIVE (after {poll_count} checks, {elapsed_time:.1f}s)")
                    return file_info
                
                elif state == 'FAILED':
                    error_msg = getattr(file_info, 'error', {}).get('message', 'Unknown error')
                    raise RuntimeError(
                        f"File {file_name} processing FAILED: {error_msg}"
                    )
                
                elif state in ('PROCESSING', 'PENDING', 'STATE_UNSPECIFIED'):
                    if poll_count == 1:
                        print(f"File {file_name} is {state}, waiting for ACTIVE state...")
                    
                    if poll_count == 1:
                        wait_time = 0.3
                    else:
                        wait_time = min(
                            poll_interval * (1.3 ** min(poll_count - 2, 4)),
                            poll_interval * 3
                        )
                    await asyncio.sleep(wait_time)
                
                else:
                    print(f"Warning: File {file_name} in unknown state: {state}. Continuing to poll...")
                    await asyncio.sleep(poll_interval)
                    
            except (TimeoutError, RuntimeError):
                raise
            except Exception as e:
                print(f"Error checking file state (attempt {poll_count}): {e}. Retrying...")
                await asyncio.sleep(poll_interval)
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcript for RAG embedding."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(pattern in line for pattern in ['[', '(', ')']):
                if line.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').replace('-', '').strip().isdigit():
                    continue
            
            if any(skip in line.lower() for skip in ['transcription:', 'transcript:', 'timestamp:', 'speaker:']):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    async def _save_transcript(
        self,
        transcript: str,
        video_path: str,
        output_dir: Optional[str] = None
    ) -> str:
        """Save transcript to file."""
        if output_dir is None:
            video_dir = Path(video_path).parent
            output_dir = str(video_dir / "transcripts")
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / f"{video_name}_transcript.txt"
        
        await asyncio.to_thread(
            lambda: output_path.write_text(transcript, encoding='utf-8')
        )
        
        return str(output_path)
    
    async def transcribe_video(
        self, 
        video_path: str, 
        prompt: Optional[str] = None,
        include_timestamps: bool = False,
        poll_interval: float = 0.5,
        max_wait_time: float = 300.0,
        initial_delay: float = 0.0,
        save_to_file: bool = True,
        output_dir: Optional[str] = None,
        clean_for_rag: bool = True
    ) -> str:
        """Transcribe video using Gemini API."""
        await self._verify_file_exists(video_path)
        
        if prompt is None:
            if include_timestamps:
                transcription_prompt = "Transcribe speech. Include timestamps."
            else:
                transcription_prompt = "Transcribe speech only."
        else:
            transcription_prompt = prompt
        
        uploaded_file = await self._run_in_executor(
            partial(self.client.files.upload, file=video_path)
        )
        
        file_name = uploaded_file.name
        
        try:
            active_file = await self._wait_for_file_active(
                file_name,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time,
                initial_delay=initial_delay
            )
            
            response = await self._run_in_executor(
                partial(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=[transcription_prompt, active_file]
                )
            )
            
            transcription = response.text if hasattr(response, 'text') else str(response)
            
            if clean_for_rag:
                transcription = self._clean_transcript(transcription)
            
            if save_to_file:
                transcript_path = await self._save_transcript(
                    transcription, video_path, output_dir
                )
                print(f"Transcript saved to: {transcript_path}")
            
            return transcription
            
        finally:
            asyncio.create_task(self._cleanup_file(file_name))
    
    async def _cleanup_file(self, file_name: str) -> None:
        """Cleanup uploaded file."""
        try:
            await self._run_in_executor(
                partial(self.client.files.delete, name=file_name)
            )
            self._uploaded_files.pop(file_name, None)
            self._file_state_cache.pop(file_name, None)
        except Exception as e:
            print(f"Warning: Failed to cleanup file {file_name}: {e}")
    
    async def transcribe_with_metadata(
        self, 
        video_path: str,
        prompt: Optional[str] = None,
        include_timestamps: bool = False,
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe video with metadata."""
        file_size_task = asyncio.create_task(self._get_file_size(video_path))
        transcription_task = asyncio.create_task(
            self.transcribe_video(
                video_path, prompt, include_timestamps,
                save_to_file=save_to_file, output_dir=output_dir
            )
        )
        
        file_size, transcription = await asyncio.gather(
            file_size_task,
            transcription_task
        )
        
        metadata = {
            "transcription": transcription,
            "video_path": str(Path(video_path).resolve()),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "model": self.model,
        }
        
        return metadata
    
    async def transcribe_multiple(
        self, 
        video_paths: list[str],
        prompt: Optional[str] = None,
        include_timestamps: bool = False,
        max_concurrent: int = 3,
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Transcribe multiple videos concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_with_semaphore(video_path: str) -> tuple[str, str]:
            async with semaphore:
                transcription = await self.transcribe_video(
                    video_path, prompt, include_timestamps,
                    save_to_file=save_to_file, output_dir=output_dir
                )
                return video_path, transcription
        
        tasks = [
            transcribe_with_semaphore(video_path) 
            for video_path in video_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        transcriptions = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Error in transcription: {result}")
                continue
            video_path, transcription = result
            transcriptions[video_path] = transcription
        
        return transcriptions
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        cleanup_tasks = [
            self._cleanup_file(file_name) 
            for file_name in list(self._uploaded_files.keys())
        ]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        if self._own_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        else:
            self._release_executor()
        
        return False


async def transcribe_video_async(
    video_path: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    prompt: Optional[str] = None,
    include_timestamps: bool = False,
    save_to_file: bool = True,
    output_dir: Optional[str] = None
) -> str:
    """Convenience function for async video transcription."""
    async with AsyncVideoTranscriber(api_key=api_key, model=model) as transcriber:
        return await transcriber.transcribe_video(
            video_path, prompt, include_timestamps,
            save_to_file=save_to_file, output_dir=output_dir
        )


class VideoTranscriber:
    """Synchronous wrapper for AsyncVideoTranscriber."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self._async_transcriber = AsyncVideoTranscriber(api_key=api_key, model=model)
    
    def transcribe_video(
        self, 
        video_path: str, 
        prompt: Optional[str] = None,
        include_timestamps: bool = False,
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> str:
        """Synchronous transcription."""
        return asyncio.run(
            self._async_transcriber.transcribe_video(
                video_path, prompt, include_timestamps,
                save_to_file=save_to_file, output_dir=output_dir
            )
        )
    
    def transcribe_with_metadata(
        self, 
        video_path: str,
        prompt: Optional[str] = None,
        include_timestamps: bool = False,
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous transcription with metadata."""
        return asyncio.run(
            self._async_transcriber.transcribe_with_metadata(
                video_path, prompt, include_timestamps,
                save_to_file=save_to_file, output_dir=output_dir
            )
        )


async def main():
    """Example usage of async transcriber."""
    video_path = os.path.join(
        Path(__file__).parent.parent.parent.parent, 
        "YC_Founder_Vid.mp4"
    )
    
    if not os.path.exists(video_path):
        video_path = "../../../YC_Founder_Vid.mp4"
    
    print(f"Transcribing video: {video_path}")
    
    async with AsyncVideoTranscriber() as transcriber:
        transcription = await transcriber.transcribe_video(video_path)
        print("\n=== Transcription ===\n")
        print(transcription)


if __name__ == "__main__":
    asyncio.run(main())