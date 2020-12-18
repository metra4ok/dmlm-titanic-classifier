from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json



class InvalidInputError(Exception):
    """Invalid model input."""


class TitanicDataRequestSchema(Schema):
    pclass = fields.Integer()
    sex = fields.Str(allow_none=True)
    age = fields.Float(allow_none=True)
    sibsp = fields.Float()
    parch = fields.Float()
    fare = fields.Float(allow_none=True)
    cabin = fields.Str(allow_none=True)
    embarked = fields.Str(allow_none=True)
    title = fields.Str(allow_none=True)


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = TitanicDataRequestSchema(strict=True, many=True)

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors

