# Cognitive-Q-Learning
Cognitive Q-Learning algorithms for team capture the flag

To train two teams playing against each other, simply execute the following command:

python cooperative_capture.py --environment [env] --blue_team [type] --red_team [type]

The argument [env] is replaced by CTF_V1.

Argument [type] is replaced by a type of algorithms: cogQL_none for cognitive MA Q-learning, cogQL_leniency for lenient cognitive MA Q-learning, cogQL_hysteretic for hysteretic cognitive MA Q-learning, QL__leniency for lenient MA Q-learning, QL_hysteretic for hysteretic MA Q-learning, QL_none for MA Q-learning.
