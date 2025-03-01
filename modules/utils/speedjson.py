import typing

try:
    import orjson
    import json
except ImportError:
    orjson = None
    import json


def load(
    fp,
    *,
    cls=None,
    object_hook=None,
    parse_float=None,
    parse_int=None,
    parse_constant=None,
    object_pairs_hook=None,
    **kw,
)-> typing.Any:
    return loads(
        fp.read(),
        cls=cls,
        object_hook=object_hook,
        parse_float=parse_float,
        parse_int=parse_int,
        parse_constant=parse_constant,
        object_pairs_hook=object_pairs_hook,
        **kw,
    )


def loads(
    s: str | bytes | bytearray,
    *,
    cls=None,
    object_hook=None,
    parse_float=None,
    parse_int=None,
    parse_constant=None,
    object_pairs_hook=None,
    **kw,
):
    if orjson and not cls:
        return orjson.loads(s)
    else:
        json.loads(
            s,
            cls=cls,
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            object_pairs_hook=object_pairs_hook,
        )


def dumps(
    obj,
    *,
    skipkeys=False,
    ensure_ascii=True,
    check_circular=True,
    allow_nan=True,
    cls=None,
    indent=None,
    separators=None,
    default=None,
    sort_keys=False,
    **kw,
) -> str:
    kwargs = {
        "skipkeys": [False, skipkeys],
        "ensure_ascii": [True, ensure_ascii],
        "check_circular": [True, check_circular],
        "allow_nan": [True, allow_nan],
        "cls": [None, cls],
        "indent": [None, indent],
        "separators": [None, separators],
        "default": [None, default],
        "sort_keys": [False, sort_keys],
    }
    # Extract non default arguments
    non_default = dict(
        [
            (arguments[0], arguments[1][0] != arguments[1][1])
            for arguments in kwargs.items()
        ]
    )
    if orjson:
        orjson_flags = 0
        if non_default:
            # Special case some arguments that is discardable.
            if "ensure_ascii" in non_default:
                non_default.pop("ensure_ascii")
            if "sort_keys" in non_default:
                non_default.pop("sort_keys")
                orjson_flags |= orjson.OPT_SORT_KEYS
            if "indent" in non_default:
                orjson_flags |= orjson.OPT_INDENT_2
        kwargs_not_default = non_default and not kw
        if not kwargs_not_default:
            return orjson.dumps(obj, option=orjson_flags).decode("utf-8")
    return json.dumps(
        obj,
        skipkeys=skipkeys,
        ensure_ascii=ensure_ascii,
        check_circular=check_circular,
        allow_nan=allow_nan,
        cls=cls,
        indent=indent,
        separators=separators,
        default=default,
        sort_keys=sort_keys,
        **kw,
    )
