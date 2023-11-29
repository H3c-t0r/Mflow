import re
from typing import List, Union

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

VALID_CONTENT_TYPE_CHARS = re.compile(r"^[a-zA-Z0-9\-/;,+_ ]*$")


def _validate_content_type(flask_request, allowed_content_types: Union[str, List[str]]):
    """
    Validates that the request content type is one of the allowed content types.

    :param flask_request: Flask request object (flask.request)
    :param allowed_content_types: Allowed content types, either a single content type or a list.
    """
    if flask_request.method not in ["POST", "PUT"]:
        return

    if isinstance(allowed_content_types, str):
        allowed_content_types = [allowed_content_types]

    # Remove any parameters e.g. "application/json; charset=utf-8" -> "application/json"
    content_type = flask_request.content_type.split(";")[0] if flask_request.content_type else None
    if content_type not in allowed_content_types:
        message = f"Bad Request. Content-Type must be one of [{', '.join(allowed_content_types)}]."

        # Avoid XSS by restricting to a set of characters
        if isinstance(content_type, str) and VALID_CONTENT_TYPE_CHARS.match(content_type):
            message += f" Got '{content_type}'."

        raise MlflowException(
            message=message,
            error_code=INVALID_PARAMETER_VALUE,
        )
