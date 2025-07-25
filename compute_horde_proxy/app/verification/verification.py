from app.client.compute_horde_client import ComputeHordeVLLMClient
from app.client.client_selector import get_or_create_client
from loguru import logger
import json
import os
import pathlib
import time
import uuid

async def run_trusted_verification(request):
    """
    Run verification using a trusted miner.

    This function is called periodically to verify the responses from compute horde miners.
    """
    try:
        # Check if it's time to run trusted verification
        if not await request.app.state.request_log.should_run_trusted_verification():
            return

        logger.info("Running trusted verification")

        # Get records that need verification
        records = await request.app.state.request_log.get_records_for_verification()

        if not records:
            logger.info("No records to verify")
            return

        for record in records:
            try:
                # Create a trusted client
                trusted_client = await ComputeHordeVLLMClient.create(
                    record.model_id, trusted=True
                )

                # Send the request to the trusted client
                trusted_response = await trusted_client.proxy_request(
                    f"/{record.endpoint}", record.method, record.request_body
                )

                # Compare the responses
                match = await request.app.state.request_log.compare_responses(
                    record.endpoint, record.response, trusted_response
                )

                # Update the verification status
                await request.app.state.request_log.update_verification_status(
                    record.id, "verified" if match else "failed"
                )

                # Log the comparison result
                await request.app.state.request_log.log_comparison_result(
                    record.id, record.model_id, record.endpoint, match
                )

                if not match:
                    logger.warning(f"Trusted verification failed for {record.model_id} at {record.endpoint}")
            except Exception as e:
                logger.error(f"Error in trusted verification for record {record.id}: {e}")
    except Exception as e:
        logger.error(f"Error in trusted verification: {e}")

async def verify_with_compute_horde_client(request, path, method, body, model_id, record_id=None):
    """
    Send a request to a compute horde client for verification and compare the response.

    Args:
        request: The FastAPI request object.
        path: The endpoint path.
        method: The HTTP method.
        body: The request body.
        model_id: The model ID.
        record_id: The ID to use for the record, to match with the local record.
    """
    try:
        # Get or create a compute horde client using local=False
        compute_horde_client = await get_or_create_client(request, model_id, local=False)

        # Send the request to the compute horde client
        compute_horde_response = await compute_horde_client.proxy_request(f"/{path}", method, body)

        # Save the request and response as JSON
        # Create directories if they don't exist
        log_dir = pathlib.Path("request_logs/compute_horde")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a JSON file with the record data
        record_data = {
            "id": record_id,
            "timestamp": time.time(),
            "model_id": model_id,
            "endpoint": path,
            "method": method,
            "request_body": body,
            "response": compute_horde_response,
            "client_type": "compute_horde"
        }

        # Save the JSON file
        file_path = log_dir / f"{record_id}.json"
        with open(file_path, 'w') as f:
            json.dump(record_data, f, indent=2)

        logger.debug(f"ComputeHordeVLLMClient response {file_path}")

    except Exception as e:
        logger.error(f"Error in parallel verification: {e}")
