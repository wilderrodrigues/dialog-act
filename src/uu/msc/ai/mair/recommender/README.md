# Restaurant Recommender System

The functions below live in [functions.py](./core/functions.py). They are pure functions: anything that has to be
remembered between turns, such as the remaining alternatives, is kept by the caller.


## Dialog manager

The state transition diagram is implemented in [dialog_manager.py](./core/dialog_manager.py). The core is
`DialogManager.transition(state, act, utterance, context)`, which returns the next state and the system utterance.
The user utterance is classified with the rule-based classifier of Part 1a; preferences are extracted with
`keyword_matching`, restaurants are found with `find_restaurants` and all responses come from `data/templates.json`.

```sh
poetry run dialog-acts chat
poetry run dialog-acts chat --show-acts   # also print the classified act, state and preferences per turn
```
