TEST_CASES = [
    {
        "question": "What does %DATE do in IBMi RPG?",
        "must_contain": [
            "%DATE",
            "date"
        ],
        "must_not_contain": [
            "%TIME",
            "%TIMESTAMP"
        ],
        "ideal_answer": (
            "%DATE returns the current system date or converts a value "
            "to date format. When called without parameters it returns "
            "today's date."
        )
    },
    {
        "question": "What does %PARMS do in IBMi RPG?",
        "must_contain": [
            "%PARMS",
            "number of parameters",
            "procedure"
        ],
        "must_not_contain": [
            "%DATE",
            "%LEN"
        ],
        "ideal_answer": (
            "%PARMS returns the number of parameters passed to the "
            "current procedure. Used to check if optional parameters "
            "were provided."
        )
    },
    {
        "question": "What does %LEN do in IBMi RPG?",
        "must_contain": [
            "%LEN",
            "length",
        ],
        "must_not_contain": [
            "%DATE",
            "%PARMS"
        ],
        "ideal_answer": (
            "%LEN returns the declared length of a field, string, array, "
            "or data structure. For character fields it returns the number "
            "of characters. For arrays it returns the number of elements. "
            "For data structures it returns the total length in bytes."
        )
    },
    {
        "question": "What is the capital of Australia?",
        "must_contain": [
            "Canberra"
        ],
        "must_not_contain": [],
        "ideal_answer": "The capital of Australia is Canberra."
    }
]