"""Dependency-free action cadence shared by runtime and supervisor prompts."""


def monitoring_config(config):
    """Preserve legacy four-action limits when no cadence was recorded."""
    maximum = config.get('max_actions_per_decision', 4)
    interval = config.get('high_level_monitor_interval_actions', maximum)
    if (type(maximum) is not int or type(interval) is not int
            or maximum < 1 or interval < 1 or maximum != interval):
        raise ValueError('Monitoring interval and action batch limit must be equal positive integers')
    result = dict(max_actions_per_decision=maximum, high_level_monitor_interval_actions=interval)
    if 'max_takeover_actions_per_decision' in config:
        takeover = config['max_takeover_actions_per_decision']
        if type(takeover) is not int or takeover < 1:
            raise ValueError('Takeover action batch limit must be a positive integer')
        result['max_takeover_actions_per_decision'] = takeover
    return result


def decision_action_limit(config, mode):
    cadence = monitoring_config(config)
    if mode == 'takeover':
        return cadence.get('max_takeover_actions_per_decision', cadence['max_actions_per_decision'])
    return cadence['max_actions_per_decision']
