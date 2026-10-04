"""Drop-in json replacement backed by orjson."""
import orjson

JSONDecodeError = orjson.JSONDecodeError


def loads(s, **_):
    return orjson.loads(s)


def dumps(obj, *, default=None, indent=None, sort_keys=False, **_):
    option = 0
    if indent is not None:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    return orjson.dumps(obj, default=default, option=option or None).decode()


def load(fp, **_):
    return orjson.loads(fp.read())


def dump(obj, fp, *, default=None, indent=None, sort_keys=False, **_):
    option = 0
    if indent is not None:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    fp.write(orjson.dumps(obj, default=default, option=option or None).decode())
