NO_PLAN = {
    "planner_class": "no_plan",
    "storage_dir": "output/plans",
}

ALL_PLAN_CONFIGS = {
    "no_plan": NO_PLAN,
}


def is_plan_training_sequential(plan_type: str) -> bool:
    return False
