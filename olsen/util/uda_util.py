import math


def get_tsa_thresh(schedule, global_step, num_train_steps, num_classes):
    start = 1.0 / num_classes
    end = 1
    progress = min(1.0, float(global_step) / float(num_train_steps))
    training_progress = progress
    scale = 5
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        threshold = math.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        threshold = 1 - math.exp((-training_progress) * scale)
    elif schedule == 'root_schedule':
        threshold = math.pow(training_progress, float(1 / scale))
    elif schedule == 'shifted_schedule':
        threshold = 0.5 + training_progress / 2
    else:
        raise ValueError
    return threshold * (end - start) + start
