with open("src/planner/engine.py", "r") as f:
    text = f.read()

import re

# Patch _scan_for_windows bare except
text = text.replace(
    "except Exception:\n            dvs.append(1e9)",
    "except Exception as e:\n            import logging\n            logging.getLogger('tars.planner').debug('Scan window error: %s', e)\n            dvs.append(1e9)"
)

# Patch _eval_direct bare except
text = text.replace(
    "except Exception:\n                pass",
    "except Exception as e:\n                import logging\n                logging.getLogger('tars.planner').debug('Direct eval error: %s', e)\n                pass"
)

# Patch _eval_multileg bare except
text = text.replace(
    "except Exception:\n        return None",
    "except Exception as e:\n        import logging\n        logging.getLogger('tars.planner').debug('Multileg eval error: %s', e)\n        return None"
)

with open("src/planner/engine.py", "w") as f:
    f.write(text)
