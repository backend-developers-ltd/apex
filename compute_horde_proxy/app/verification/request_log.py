import random
import uuid
import time
import json
import pathlib
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
import asyncio
from loguru import logger
import os

class RequestRecord(BaseModel):
    """Record of a request and its response."""
    id: str
    timestamp: float
    model_id: str
    endpoint: str
    method: str
    request_body: Optional[Dict[str, Any]] = None
    response: Dict[str, Any]
    miner_id: Optional[str] = None  # UUID of the compute horde miner
    client_type: str  # "base" or "compute_horde"
    verification_status: Optional[str] = None  # "pending", "verified", "failed"

class RequestLog:
    """Log of requests and responses for verification."""
    def __init__(self, max_size: int = 100, sample_rate: float = 0.1):
        self.records: List[RequestRecord] = []
        self.max_size = max_size
        self.sample_rate = sample_rate
        self.lock = asyncio.Lock()
        self.verification_enabled = os.environ.get("ENABLE_VERIFICATION", "false").lower() == "true"
        self.parallel_verification = os.environ.get("ENABLE_PARALLEL_VERIFICATION", "false").lower() == "true"
        self.trusted_verification_interval = int(os.environ.get("TRUSTED_VERIFICATION_INTERVAL", "3600"))  # Default: 1 hour
        self.last_trusted_verification = 0

        # Create directories for storing request/response data
        self.base_log_dir = pathlib.Path(os.environ.get("REQUEST_LOG_DIR", "request_logs"))
        self.base_client_log_dir = self.base_log_dir / "base_client"
        self.compute_horde_log_dir = self.base_log_dir / "compute_horde_client"
        self.comparison_log_file = self.base_log_dir / "comparison_results.log"

        # Create directories if they don't exist
        self.base_client_log_dir.mkdir(parents=True, exist_ok=True)
        self.compute_horde_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize comparison log file if it doesn't exist
        if not self.comparison_log_file.exists():
            with open(self.comparison_log_file, 'w') as f:
                f.write("timestamp,request_id,model_id,endpoint,comparison_result\n")

    async def add_record(self, model_id: str, endpoint: str, method: str, request_body: Dict[str, Any], 
                         response: Dict[str, Any], client_type: str, miner_id: Optional[str] = None, 
                         record_id: Optional[str] = None) -> Optional[str]:
        """
        Add a record to the log and save it to a JSON file.

        Args:
            model_id: The ID of the model.
            endpoint: The endpoint that was called.
            method: The HTTP method that was used.
            request_body: The request body.
            response: The response from the endpoint.
            client_type: The type of client ("base" or "compute_horde").
            miner_id: The UUID of the compute horde miner (only for compute_horde client).
            record_id: Optional record ID to use. If not provided, a new UUID will be generated.

        Returns:
            The record ID if the record was added, None otherwise.
        """
        # Only sample a percentage of requests
        if random.random() > self.sample_rate and record_id is None:
            return None

        async with self.lock:
            # Create a new record ID if not provided
            if record_id is None:
                record_id = str(uuid.uuid4())

            # Create a new record
            record = RequestRecord(
                id=record_id,
                timestamp=time.time(),
                model_id=model_id,
                endpoint=endpoint,
                method=method,
                request_body=request_body,
                response=response,
                miner_id=miner_id,
                client_type=client_type,
                verification_status="pending" if self.verification_enabled else None
            )

            # Add the record to the log
            self.records.append(record)

            # If the log is too large, remove the oldest records
            if len(self.records) > self.max_size:
                self.records = self.records[-self.max_size:]

            # Save the record to a JSON file
            try:
                # Determine the log directory based on client type
                log_dir = self.base_client_log_dir if client_type == "base" else self.compute_horde_log_dir

                # Create a JSON file with the record data
                record_data = {
                    "id": record_id,
                    "timestamp": record.timestamp,
                    "model_id": model_id,
                    "endpoint": endpoint,
                    "method": method,
                    "request_body": request_body,
                    "response": response,
                    "miner_id": miner_id,
                    "client_type": client_type,
                    "verification_status": record.verification_status
                }

                # Save the JSON file
                file_path = log_dir / f"{record_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(record_data, f, indent=2)

                logger.debug(f"Saved request/response record to {file_path}")
            except Exception as e:
                logger.error(f"Error saving request/response record: {e}")

            return record_id

    async def get_record(self, record_id: str) -> Optional[RequestRecord]:
        """Get a record by ID."""
        async with self.lock:
            for record in self.records:
                if record.id == record_id:
                    return record
            return None

    async def update_verification_status(self, record_id: str, status: str):
        """Update the verification status of a record."""
        async with self.lock:
            for record in self.records:
                if record.id == record_id:
                    record.verification_status = status
                    break

    async def get_records_for_verification(self, limit: int = 10) -> List[RequestRecord]:
        """Get records that need verification."""
        async with self.lock:
            pending_records = [r for r in self.records if r.verification_status == "pending"]
            return pending_records[:limit]

    async def should_run_trusted_verification(self) -> bool:
        """Check if it's time to run trusted verification."""
        if not self.verification_enabled:
            return False

        current_time = time.time()
        if current_time - self.last_trusted_verification > self.trusted_verification_interval:
            self.last_trusted_verification = current_time
            return True
        return False

    async def log_comparison_result(self, request_id: str, model_id: str, endpoint: str, result: bool):
        """
        Log the comparison result to the comparison log file.

        Args:
            request_id: The ID of the request.
            model_id: The ID of the model.
            endpoint: The endpoint that was called.
            result: The comparison result (True if responses match, False otherwise).
        """
        try:
            timestamp = time.time()
            result_str = "match" if result else "mismatch"
            log_entry = f"{timestamp},{request_id},{model_id},{endpoint},{result_str}\n"

            with open(self.comparison_log_file, 'a') as f:
                f.write(log_entry)

            logger.debug(f"Logged comparison result for request {request_id}: {result_str}")
        except Exception as e:
            logger.error(f"Error logging comparison result: {e}")

    async def compare_responses(self, endpoint: str, response1: Dict[str, Any], response2: Dict[str, Any]) -> bool:
        """
        Compare two responses based on the endpoint type.

        Returns:
            True if the responses match, False otherwise.
        """
        try:
            if "generate" in endpoint and "logits" not in endpoint:
                # For generate endpoints, compare the text content
                if "choices" in response1 and "choices" in response2:
                    text1 = response1["choices"][0]["message"]["content"]
                    text2 = response2["choices"][0]["message"]["content"]
                    # Simple string comparison for now
                    return text1 == text2
            elif "generate_logits" in endpoint:
                # For generate_logits endpoints, compare the logprobs
                if "logprobs" in response1 and "logprobs" in response2:
                    # Compare top tokens
                    tokens1 = [item["token"] for item in response1["logprobs"][0]["top_logprobs"]]
                    tokens2 = [item["token"] for item in response2["logprobs"][0]["top_logprobs"]]
                    return tokens1[:5] == tokens2[:5]  # Compare top 5 tokens
            elif "embeddings" in endpoint:
                # For embeddings endpoints, compare the embedding vectors
                if "data" in response1 and "data" in response2:
                    # TODO implement a comparison
                    # Simple check for now - just compare the length
                    return len(response1["data"]) == len(response2["data"])

            # Default comparison
            return response1 == response2
        except Exception as e:
            logger.error(f"Error comparing responses: {e}")
            return False
