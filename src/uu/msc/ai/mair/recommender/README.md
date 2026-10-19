# Restaurant Recommender System

The functions below live in [functions.py](./core/functions.py). They are pure functions: anything that has to be
remembered between turns, such as the remaining alternatives, is kept by the caller.


## Dialog manager

The state transition diagram is implemented in [dialog_manager.py](./core/dialog_manager.py). The core is
`DialogManager.transition(state, act, utterance, context)`, which returns the next state and the system utterance.
The `chat` command uses the trained logistic-regression dialog-act classifier. Each state handler uses that act to
decide whether the utterance is valid for the current step; slot and requirement extraction then determines which
decision in the flow chart to take. Invalid or unrecognized inputs are explained and reprompted without changing the
dialog context. The manager also accepts preferences and requirements in the opening utterance, and allows preference
changes after a recommendation. Preferences are extracted with `keyword_matching` and the configured fallback,
restaurants are found with `find_restaurants`, and responses come from `data/templates.json`.

```sh
poetry run dialog-acts chat
poetry run dialog-acts chat --show-acts   # also print the classified act, state and preferences per turn
```

## Reasoning

After the three preferences are known, the system asks for additional requirements (state 5): touristic, assigned
seats, children or romantic. [reasoning.py](./core/reasoning.py) applies the six inference rules to every restaurant
that matches the preferences and keeps those whose derived properties match the requirements. This needs
`restaurant_info_extended.csv`, which is the default of the `chat` command.

When rules contradict each other (e.g. busy -> not romantic, long stay -> romantic) the property stays unknown, so the
restaurant is not recommended for that requirement. If that leaves no restaurant, the system names the closest one and
explains which rules contradict each other.

```sh
poetry run dialog-acts chat --no-transparent   # hide the reasoning behind a recommendation
```
