import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter


def _is_pytest_raises_call(node: astroid.NodeNG):
    if not isinstance(node, astroid.Call):
        return False
    if not isinstance(node.func, astroid.Attribute) or not isinstance(node.func.expr, astroid.Name):
        return False
    return node.func.expr.name == "pytest" and node.func.attrname == "raises"


def _called_with_match(node: astroid.Call):
    # Note `match` is a keyword-only argument:
    # https://docs.pytest.org/en/latest/reference/reference.html#pytest.raises
    return any(k.arg == "match" for k in node.keywords)


def _is_complex_pytest_raises(raises_with: astroid.With):
    if len(raises_with.body) > 1:
        return True

    if isinstance(
        raises_with.body[0],
        (
            astroid.If,
            astroid.For,
            astroid.While,
            astroid.TryExcept,
            astroid.TryFinally,
        ),
    ):
        return True

    # Nested with
    if isinstance(raises_with.body[0], astroid.With):
        nested_with = raises_with.body[0]
        if len(nested_with.body) > 1 or not isinstance(nested_with.body[0], astroid.Pass):
            return True

    return False


class PytestRaisesChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "pytest-raises-checker"
    WITHOUT_MATCH = "pytest-raises-without-match"
    COMPLEX_BODY = "pytest-raises-complex-body"
    msgs = {
        "W0001": (
            "`pytest.raises` must be called with `match` argument`",
            WITHOUT_MATCH,
            "Use `pytest.raises(<exception>, match=...)`",
        ),
        "W0004": (
            "`pytest.raises` block should not contain multiple statements and control flow"
            " structures. It should only contain a single statement that throws an exception.",
            COMPLEX_BODY,
            "Any initialization/finalization code should be moved outside of `pytest.raises` block",
        ),
    }
    priority = -1

    def __init__(self, linter: PyLinter) -> None:
        super().__init__(linter)
        self._is_in_pytest_raises = False

    def visit_call(self, node: astroid.Call):
        if not _is_pytest_raises_call(node):
            return

        if not _called_with_match(node):
            self.add_message(PytestRaisesChecker.WITHOUT_MATCH, node=node)

    def visit_assert(self, node: astroid.Assert):
        if self._is_in_pytest_raises:
            self.add_message(PytestRaisesChecker.CONTAINS_ASSERTIONS, node=node)

    def visit_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items) and (
            _is_complex_pytest_raises(node)
        ):
            self.add_message(PytestRaisesChecker.COMPLEX_BODY, node=node)

    def leave_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items):
            self._is_in_pytest_raises = False
