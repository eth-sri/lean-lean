from leanlean import Planner, Environment, Model, Instance

class NoPlanPlanner(Planner):
    """A planner that does nothing."""

    def __init__(self, **kwargs) -> None:
        pass

    def plan(self, env: Environment, model: Model, instance: Instance) -> None:
        return

    def update_plan(self, **kwargs) -> None:
        pass
