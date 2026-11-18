import heapq
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .errors import Errors


DEPENDENCY_KEYS = ("before", "after")


def normalize_dependencies(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Normalize a pipeline dependency declaration to a tuple of names."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(Errors.E1060.format(value=repr(value)))
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def normalize_pipe_dependencies(
    *,
    before: Optional[Iterable[str]] = None,
    after: Optional[Iterable[str]] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Create a normalized dependency dict for one pipeline component."""
    return {
        "before": normalize_dependencies(before),
        "after": normalize_dependencies(after),
    }


def extract_pipe_dependencies(
    config: Mapping[str, Any],
    *,
    factory_args: Iterable[str] = (),
) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, ...]], Set[str]]:
    """Remove dependency settings from a component config.

    Both the flat form (``before = ["a"]``) and the nested form
    (``dependencies = {before = ["a"]}``) are supported. The returned config
    can safely be passed to the component factory. Flat keys that are already
    accepted by the factory are preserved to keep existing configs compatible.
    """
    config = dict(config)
    factory_args = set(factory_args)
    dependencies: Dict[str, Tuple[str, ...]] = {}
    stripped: Set[str] = set()
    nested = (
        config.pop("dependencies")
        if "dependencies" in config and "dependencies" not in factory_args
        else None
    )
    if nested is not None:
        stripped.add("dependencies")
        if isinstance(nested, (str, list, tuple)):
            nested = {"after": nested}
        if not isinstance(nested, dict):
            raise ValueError(Errors.E1061.format(value=repr(nested)))
        invalid = set(nested) - set(DEPENDENCY_KEYS)
        if invalid:
            opts = ", ".join(sorted(DEPENDENCY_KEYS))
            raise ValueError(Errors.E1062.format(key=sorted(invalid)[0], opts=opts))
        for key in DEPENDENCY_KEYS:
            if key in nested:
                dependencies[key] = normalize_dependencies(nested[key])
    for key in DEPENDENCY_KEYS:
        if key in config and key not in factory_args:
            dependencies[key] = normalize_dependencies(config.pop(key))
            stripped.add(key)
    return config, dependencies, stripped


def sort_pipeline_components(
    component_names: Sequence[str],
    factories: Mapping[str, str],
    dependencies: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    validate: bool = True,
) -> List[str]:
    """Sort pipeline component names according to before/after dependencies.

    Dependencies can refer to a component instance name or a factory name. A
    factory reference applies to all instances of that factory. The original
    component order is used as a stable tie-breaker, so components without
    constraints are never reordered.
    """
    names = list(component_names)
    name_set = set(names)
    names_by_factory: Dict[str, Set[str]] = {}
    for name in names:
        factory = factories.get(name)
        if factory is not None:
            names_by_factory.setdefault(factory, set()).add(name)
    graph: Dict[str, Set[str]] = {name: set() for name in names}
    unresolved: List[Tuple[str, str, str]] = []

    def resolve(name: str, ref: str) -> Set[str]:
        if ref in name_set:
            return {ref}
        # A factory dependency applies to all other instances of that factory.
        # Excluding the current component avoids an accidental self-cycle when
        # a differently named instance refers to its own factory.
        return names_by_factory.get(ref, set()) - {name}

    for name in names:
        component_deps = dependencies.get(name, {})
        for ref in component_deps.get("after", ()):
            targets = resolve(name, ref)
            if not targets:
                unresolved.append((name, "after", ref))
            for target in targets:
                graph[target].add(name)
        for ref in component_deps.get("before", ()):
            targets = resolve(name, ref)
            if not targets:
                unresolved.append((name, "before", ref))
            for target in targets:
                graph[name].add(target)

    if unresolved and validate:
        component, direction, ref = unresolved[0]
        opts = ", ".join(names)
        raise ValueError(
            Errors.E1058.format(
                component=component,
                direction=direction,
                name=ref,
                opts=opts,
            )
        )

    indegree = dict.fromkeys(names, 0)
    for targets in graph.values():
        for target in targets:
            indegree[target] += 1
    positions = {name: i for i, name in enumerate(names)}
    queue = [(positions[name], name) for name in names if indegree[name] == 0]
    heapq.heapify(queue)
    result = []
    while queue:
        _, name = heapq.heappop(queue)
        result.append(name)
        for target in sorted(graph[name], key=positions.__getitem__):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(queue, (positions[target], target))
    if len(result) != len(names):
        cycle = _find_cycle(graph, [name for name in names if indegree[name] > 0])
        raise ValueError(Errors.E1059.format(cycle=" -> ".join(cycle)))
    return result


def _find_cycle(graph: Mapping[str, Set[str]], remaining: Iterable[str]) -> List[str]:
    """Return one dependency cycle as a path, including the repeated start."""
    remaining_set = set(remaining)
    visited: Set[str] = set()
    active: List[str] = []
    active_set: Set[str] = set()

    def visit(node: str) -> Optional[List[str]]:
        if node in active_set:
            start = active.index(node)
            return active[start:] + [node]
        if node in visited:
            return None
        visited.add(node)
        active.append(node)
        active_set.add(node)
        for target in graph[node]:
            if target not in remaining_set:
                continue
            cycle = visit(target)
            if cycle is not None:
                return cycle
        active.pop()
        active_set.remove(node)
        return None

    for name in remaining:
        if name not in visited:
            cycle = visit(name)
            if cycle is not None:
                return cycle
    return list(remaining)
