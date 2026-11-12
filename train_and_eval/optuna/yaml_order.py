"""Stable presentation order for exported run YAML; preserves every value/key."""

from copy import deepcopy

# Match the established grid layout. Unknown fields remain visible at the end.
FIELD_ORDER = {(): ('config_schema_version',
      'run',
      'continuation',
      'data',
      'training',
      'environment',
      'ppo',
      'evaluation',
      'logging',
      'artifacts'),
 ('run',): ('name', 'seed'),
 ('continuation',): ('mode', 'source_run', 'checkpoint'),
 ('data',): ('path', 'train_range', 'validation_range', 'train_ratio', 'alignment'),
 ('data', 'train_range'): ('start', 'end'),
 ('data', 'validation_range'): ('start', 'end'),
 ('training',): ('duration_unit', 'duration_amount'),
 ('environment',): ('window',
                    'context',
                    'position_side',
                    'market_timezone',
                    'rth_open',
                    'rth_close',
                    'stake_pln',
                    'fee_bps',
                    'swap_long_bps',
                    'swap_short_bps',
                    'swap_time',
                    'swap_timezone',
                    'force_close_on_done',
                    'reward_scale',
                    'exposure_penalty',
                    'turnover_penalty',
                    'drawdown_penalty',
                    'profit_reward_mult',
                    'loss_reward_mult'),
 ('ppo',): ('policy',
            'device',
            'hidden_sizes',
            'activation',
            'n_steps',
            'batch_size',
            'n_epochs',
            'learning_rate',
            'gamma',
            'gae_lambda',
            'clip_range',
            'clip_range_vf',
            'normalize_advantage',
            'ent_coef',
            'vf_coef',
            'max_grad_norm',
            'target_kl',
            'value_head_init_scale',
            'initial_long_probability'),
 ('evaluation',): ('eval_every_steps',
                   'checkpoint_every_steps',
                   'policy_mode',
                   'threshold_action',
                   'probability_threshold',
                   'best_metric',
                   'early_stop_patience_evals',
                   'device',
                   'training_mode'),
 ('logging',): ('training_progress_every_steps', 'validation_progress_every_steps'),
 ('artifacts',): ('training_metrics', 'evaluation_trajectory', 'plots'),
 ('artifacts', 'training_metrics'): ('enabled', 'every_steps'),
 ('artifacts', 'evaluation_trajectory'): ('mode',),
 ('artifacts', 'plots'): ('during_run',)}


def ordered_run_config(config):
    """Return an independent mapping ordered for humans, without normalizing values."""
    def visit(value, path):
        if isinstance(value, dict):
            preferred = FIELD_ORDER.get(path, ())
            keys = [key for key in preferred if key in value]
            keys.extend(key for key in value if key not in preferred)
            return {key: visit(value[key], path + (key,)) for key in keys}
        return deepcopy(value)

    return visit(config, ())
