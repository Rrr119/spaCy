import pytest

from spacy.lang.en import English
from spacy.language import Language


def _make_component(name, *, before=None, after=None):
    @Language.component(name, before=before or [], after=after or [])
    def component(doc):
        return doc

    return component


def test_pipeline_dependencies_sort_after_factory_metadata():
    _make_component("dep_metadata_consumer", after=["dep_metadata_provider"])
    _make_component("dep_metadata_provider")
    nlp = Language()
    # The provider can be added later: dependency validation is deferred until
    # the pipeline can be sorted as a whole.
    nlp.add_pipe("dep_metadata_consumer")
    nlp.add_pipe("dep_metadata_provider")
    assert nlp.pipe_names == ["dep_metadata_provider", "dep_metadata_consumer"]
    assert nlp.pipe_dependencies["dep_metadata_consumer"]["after"] == [
        "dep_metadata_provider"
    ]


def test_pipeline_dependencies_before_and_stable_order():
    _make_component("dep_before_first")
    _make_component("dep_before_anchor")
    _make_component("dep_before_unrelated")
    _make_component("dep_before_component", before=["dep_before_anchor"])
    nlp = Language()
    nlp.add_pipe("dep_before_first")
    nlp.add_pipe("dep_before_anchor")
    nlp.add_pipe("dep_before_unrelated")
    nlp.add_pipe("dep_before_component")
    assert nlp.pipe_names == [
        "dep_before_first",
        "dep_before_unrelated",
        "dep_before_component",
        "dep_before_anchor",
    ]
    # Independent components retain their original relative order.
    assert nlp.pipe_names.index("dep_before_first") < nlp.pipe_names.index(
        "dep_before_unrelated"
    )


def test_pipeline_dependencies_chain():
    _make_component("dep_chain_c", after=["dep_chain_b"])
    _make_component("dep_chain_b", after=["dep_chain_a"])
    _make_component("dep_chain_a")
    nlp = Language()
    nlp.add_pipe("dep_chain_c")
    nlp.add_pipe("dep_chain_b")
    nlp.add_pipe("dep_chain_a")
    assert nlp.pipe_names == ["dep_chain_a", "dep_chain_b", "dep_chain_c"]


def test_pipeline_dependencies_can_reference_factory_with_custom_name():
    @Language.factory("dep_named_factory")
    def make_named_factory(nlp, name):
        def component(doc):
            return doc

        return component

    _make_component("dep_after_named_factory", after=["dep_named_factory"])
    nlp = Language()
    nlp.add_pipe("dep_named_factory", name="custom_named_pipe")
    nlp.add_pipe("dep_after_named_factory")
    assert nlp.pipe_names == ["custom_named_pipe", "dep_after_named_factory"]


def test_pipeline_dependencies_cycle_rolls_back_add_pipe():
    _make_component("dep_cycle_a", after=["dep_cycle_b"])
    _make_component("dep_cycle_b", after=["dep_cycle_a"])
    nlp = Language()
    nlp.add_pipe("dep_cycle_a")
    with pytest.raises(ValueError, match="Cycle detected in pipeline dependencies"):
        nlp.add_pipe("dep_cycle_b")
    assert nlp.pipe_names == ["dep_cycle_a"]
    assert "dep_cycle_b" not in nlp.component_names
    assert "dep_cycle_b" not in nlp.config["components"]


def test_pipeline_dependencies_unknown_reference():
    _make_component("dep_unknown_consumer", after=["dep_does_not_exist"])
    nlp = Language()
    nlp.add_pipe("dep_unknown_consumer")
    # Adding with an unresolved forward reference is allowed.
    nlp.add_pipe("dep_unknown_consumer", name="dep_unknown_consumer_2")
    with pytest.raises(ValueError, match="no matching component or factory"):
        nlp.sort_pipeline()


def test_pipeline_dependencies_config_roundtrip():
    _make_component("dep_config_provider")
    _make_component("dep_config_consumer")
    nlp = English()
    nlp.add_pipe("dep_config_consumer", config={"after": "dep_config_provider"})
    nlp.add_pipe("dep_config_provider")
    assert nlp.pipe_names == ["dep_config_provider", "dep_config_consumer"]
    config = nlp.config.copy()
    assert config["nlp"]["pipeline"] == [
        "dep_config_provider",
        "dep_config_consumer",
    ]
    assert config["components"]["dep_config_consumer"]["after"] == [
        "dep_config_provider"
    ]
    nlp2 = English.from_config(config)
    assert nlp2.pipe_names == ["dep_config_provider", "dep_config_consumer"]
    assert nlp2.pipe_dependencies["dep_config_consumer"]["after"] == [
        "dep_config_provider"
    ]
    assert nlp2.get_pipe_meta("dep_config_consumer").after == ("dep_config_provider",)


def test_pipeline_dependencies_nested_config():
    _make_component("dep_nested_provider")
    _make_component("dep_nested_consumer")
    nlp = Language()
    nlp.add_pipe(
        "dep_nested_consumer",
        config={"dependencies": {"after": ["dep_nested_provider"]}},
    )
    nlp.add_pipe("dep_nested_provider")
    assert nlp.pipe_names == ["dep_nested_provider", "dep_nested_consumer"]
    config = nlp.config.copy()
    assert "dependencies" not in config["components"]["dep_nested_consumer"]
    assert config["components"]["dep_nested_consumer"]["after"] == [
        "dep_nested_provider"
    ]


def test_pipeline_dependencies_from_config_cycle():
    _make_component("dep_config_cycle_a")
    _make_component("dep_config_cycle_b")
    nlp = English()
    nlp.add_pipe("dep_config_cycle_a")
    nlp.add_pipe("dep_config_cycle_b")
    config = nlp.config.copy()
    config["components"]["dep_config_cycle_a"]["after"] = ["dep_config_cycle_b"]
    config["components"]["dep_config_cycle_b"]["after"] = ["dep_config_cycle_a"]
    with pytest.raises(ValueError, match="Cycle detected in pipeline dependencies"):
        English.from_config(config)


def test_pipeline_dependencies_from_config_unknown():
    _make_component("dep_config_unknown")
    nlp = English()
    nlp.add_pipe("dep_config_unknown")
    config = nlp.config.copy()
    config["components"]["dep_config_unknown"]["after"] = ["dep_config_missing"]
    with pytest.raises(ValueError, match="no matching component or factory"):
        English.from_config(config)


def test_pipeline_dependencies_from_config_exclude_target():
    _make_component("dep_config_exclude_provider")
    _make_component("dep_config_exclude_consumer")
    nlp = English()
    nlp.add_pipe("dep_config_exclude_provider")
    nlp.add_pipe(
        "dep_config_exclude_consumer",
        config={"after": ["dep_config_exclude_provider"]},
    )
    config = nlp.config.copy()
    nlp2 = English.from_config(config, exclude=["dep_config_exclude_provider"])
    assert nlp2.pipe_names == ["dep_config_exclude_consumer"]
    assert nlp2.pipe_dependencies["dep_config_exclude_consumer"]["after"] == []


def test_pipeline_dependencies_do_not_change_legacy_config():
    _make_component("dep_legacy_a")
    _make_component("dep_legacy_b")
    nlp = English()
    nlp.add_pipe("dep_legacy_b")
    nlp.add_pipe("dep_legacy_a")
    config = nlp.config.copy()
    assert config["nlp"]["pipeline"] == ["dep_legacy_b", "dep_legacy_a"]
    assert "before" not in config["components"]["dep_legacy_a"]
    assert "after" not in config["components"]["dep_legacy_a"]
    nlp2 = English.from_config(config)
    assert nlp2.pipe_names == ["dep_legacy_b", "dep_legacy_a"]


def test_pipeline_dependencies_are_not_passed_to_factory_config():
    @Language.factory("dep_strict_factory", default_config={"value": 1})
    def make_strict_factory(nlp, name, value):
        def component(doc):
            return doc

        component.value = value
        return component

    _make_component("dep_strict_provider")
    nlp = Language()
    component = nlp.add_pipe(
        "dep_strict_factory",
        config={"value": 2, "after": ["dep_strict_provider"]},
    )
    nlp.add_pipe("dep_strict_provider")
    assert component.value == 2
    assert nlp.pipe_names == ["dep_strict_provider", "dep_strict_factory"]


def test_pipeline_dependencies_keep_legacy_factory_args_named_before_after():
    @Language.factory(
        "dep_legacy_factory_args",
        default_config={"before": "default_before", "after": "default_after"},
    )
    def make_legacy_factory_args(nlp, name, before, after):
        def component(doc):
            return doc

        component.before = before
        component.after = after
        return component

    nlp = Language()
    component = nlp.add_pipe(
        "dep_legacy_factory_args",
        config={"before": "custom_before", "after": "custom_after"},
    )
    assert component.before == "custom_before"
    assert component.after == "custom_after"
    assert nlp.pipe_dependencies["dep_legacy_factory_args"] == {
        "before": [],
        "after": [],
    }
    assert nlp.config["components"]["dep_legacy_factory_args"]["before"] == (
        "custom_before"
    )


def test_pipeline_dependencies_rename_pipe():
    _make_component("dep_rename_provider")
    _make_component("dep_rename_consumer", after=["dep_rename_provider"])
    nlp = Language()
    nlp.add_pipe("dep_rename_provider")
    nlp.add_pipe("dep_rename_consumer")
    nlp.rename_pipe("dep_rename_provider", "renamed_provider")
    assert nlp.pipe_dependencies["dep_rename_consumer"]["after"] == ["renamed_provider"]
    assert nlp.get_pipe_meta("dep_rename_consumer").after == ("renamed_provider",)
    assert nlp.config["components"]["dep_rename_consumer"]["after"] == [
        "renamed_provider"
    ]
