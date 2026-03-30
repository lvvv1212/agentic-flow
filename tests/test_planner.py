"""Tests for the planning module."""

from agentic_flow.planner import Task, Plan, TaskStatus


class TestTask:
    def test_creation(self) -> None:
        t = Task(id=1, description="Do something")
        assert t.status == TaskStatus.PENDING
        assert t.result == ""

    def test_to_dict(self) -> None:
        t = Task(id=1, description="Do something", depends_on=[])
        d = t.to_dict()
        assert d["id"] == 1
        assert d["status"] == "pending"


class TestPlan:
    def test_is_complete(self) -> None:
        plan = Plan(
            goal="test",
            tasks=[
                Task(id=1, description="a", status=TaskStatus.COMPLETED),
                Task(id=2, description="b", status=TaskStatus.COMPLETED),
            ],
        )
        assert plan.is_complete

    def test_is_not_complete(self) -> None:
        plan = Plan(
            goal="test",
            tasks=[
                Task(id=1, description="a", status=TaskStatus.COMPLETED),
                Task(id=2, description="b", status=TaskStatus.PENDING),
            ],
        )
        assert not plan.is_complete

    def test_next_task(self) -> None:
        plan = Plan(
            goal="test",
            tasks=[
                Task(id=1, description="a", status=TaskStatus.COMPLETED),
                Task(id=2, description="b", status=TaskStatus.PENDING, depends_on=[1]),
            ],
        )
        nxt = plan.next_task
        assert nxt is not None
        assert nxt.id == 2

    def test_next_task_blocked(self) -> None:
        plan = Plan(
            goal="test",
            tasks=[
                Task(id=1, description="a", status=TaskStatus.PENDING),
                Task(id=2, description="b", status=TaskStatus.PENDING, depends_on=[1]),
            ],
        )
        nxt = plan.next_task
        assert nxt is not None
        assert nxt.id == 1  # Task 1 has no deps, so it's next

    def test_summary(self) -> None:
        plan = Plan(
            goal="do stuff",
            tasks=[
                Task(id=1, description="step one", status=TaskStatus.COMPLETED),
                Task(id=2, description="step two", status=TaskStatus.PENDING),
            ],
        )
        summary = plan.summary()
        assert "do stuff" in summary
        assert "[x]" in summary
        assert "[ ]" in summary
