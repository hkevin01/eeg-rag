"""Shared eligibility predicates for local and Qdrant retrieval."""

import math
from typing import Any, Mapping


def validate_filters(filters: Mapping[str, Any]) -> None:
    """Validate equality and numeric range predicates before backend execution.

    Args:
        filters: Field names mapped to scalar values or gt/gte/lt/lte bounds.

    Raises:
        ValueError: A field or predicate cannot be represented by both backends.
    """
    for key, expected in filters.items():
        if (
            not isinstance(key, str)
            or not key
            or any(not part for part in key.split("."))
        ):
            raise ValueError("Filter fields must be nonempty dotted paths")
        if isinstance(expected, Mapping):
            if not expected or set(expected) - {"gt", "gte", "lt", "lte"}:
                raise ValueError("Only gt/gte/lt/lte range filters are supported")
            if any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
                for bound in expected.values()
            ):
                raise ValueError("Range bounds must be finite numbers")
        elif not isinstance(expected, (str, int, bool)):
            raise ValueError("Equality filters require a string, integer, or boolean")


def matches_filters(document: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """Evaluate eligibility, rejecting missing values and incompatible types.

    Args:
        document: Document payload, optionally containing nested metadata.
        filters: Validated equality or numeric range predicates.

    Returns:
        Whether every predicate is satisfied. Unqualified fields prefer metadata.
    """
    for key, expected in filters.items():
        parts = key.split(".")
        value: Any = document
        if len(parts) == 1 and key in document.get("metadata", {}):
            value = document["metadata"][key]
        else:
            for part in parts:
                value = value.get(part) if isinstance(value, Mapping) else None
        if value is None:
            return False
        if isinstance(expected, Mapping):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(value):
                return False
            for operator, bound in expected.items():
                if not {
                    "gt": value > bound,
                    "gte": value >= bound,
                    "lt": value < bound,
                    "lte": value <= bound,
                }[operator]:
                    return False
        elif type(value) is not type(expected) or value != expected:
            return False
    return True


def to_qdrant_filter(filters: Mapping[str, Any]) -> Any:
    """Translate the same eligibility policy into Qdrant conditions.

    Args:
        filters: Scalar equality or numeric range predicates.

    Returns:
        Qdrant Filter, or None when no constraints were supplied.

    Raises:
        ValueError: Unsupported filter predicates.
    """
    from qdrant_client import models

    validate_filters(filters)
    conditions = []
    for key, expected in filters.items():

        def condition(path: str, expected: Any = expected) -> Any:
            if isinstance(expected, Mapping):
                return models.FieldCondition(key=path, range=models.Range(**expected))
            return models.FieldCondition(
                key=path, match=models.MatchValue(value=expected)
            )

        if "." in key:
            conditions.append(condition(key))
        else:
            metadata_path = f"metadata.{key}"
            conditions.append(
                models.Filter(
                    should=[
                        condition(metadata_path),
                        models.Filter(
                            must=[
                                models.IsEmptyCondition(
                                    is_empty=models.PayloadField(key=metadata_path)
                                ),
                                condition(key),
                            ]
                        ),
                    ]
                )
            )
    return models.Filter(must=conditions) if conditions else None
