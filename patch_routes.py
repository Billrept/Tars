with open("src/api/routes_http.py", "r") as f:
    text = f.read()

old_block = """    try:
        first_body = _resolve_body(req.body_sequence[0])
        last_body = _resolve_body(req.body_sequence[-1])
        cache.validate_epoch_range(first_body.naif_id, dep_start_jd, dep_end_jd)
        cache.validate_epoch(last_body.naif_id, max_arr_jd)
    except EphemerisRangeError as e:"""

new_block = """    try:
        for b in req.body_sequence:
            body = _resolve_body(b)
            cache.validate_epoch_range(body.naif_id, dep_start_jd, max_arr_jd)
    except EphemerisRangeError as e:"""

text = text.replace(old_block, new_block)

with open("src/api/routes_http.py", "w") as f:
    f.write(text)
