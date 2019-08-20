from spinn import util

NUMBERS = list(range(10))

FIXED_VOCABULARY = {str(x): i + 1 for i, x in enumerate(NUMBERS)}
FIXED_VOCABULARY.update({
    util.PADDING_TOKEN: 0,
    "[MIN": len(FIXED_VOCABULARY) + 1,
    "[MAX": len(FIXED_VOCABULARY) + 2,
    "[FIRST": len(FIXED_VOCABULARY) + 3,
    "[LAST": len(FIXED_VOCABULARY) + 4,
    "[MED": len(FIXED_VOCABULARY) + 5,
    "[SM": len(FIXED_VOCABULARY) + 6,
    "[PM": len(FIXED_VOCABULARY) + 7,
	"[FLSUM": len(FIXED_VOCABULARY) + 8,
    "]": len(FIXED_VOCABULARY) + 9
})
assert len(set(FIXED_VOCABULARY.values())) == len(list(FIXED_VOCABULARY.values()))
