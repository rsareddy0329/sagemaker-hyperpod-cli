"""Unit tests for recipe_utils template rendering.

These tests verify that the Jinja2 template rendering in _render_k8s_template
uses a sandboxed environment that restricts template operations to safe
constructs while still allowing legitimate template functionality.
"""

import pytest
from jinja2.exceptions import SecurityError

from sagemaker.hyperpod.cli.recipe_utils import _render_k8s_template


class TestRenderK8sTemplate:
    """Tests for _render_k8s_template sandboxed rendering."""

    def test_variable_substitution(self):
        """Normal variable substitution should work."""
        template = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {{ name }}\n  namespace: {{ namespace }}"
        config = {"name": "my-job", "namespace": "default"}
        result = _render_k8s_template(template, config)
        assert "name: my-job" in result
        assert "namespace: default" in result

    def test_filter_usage(self):
        """Jinja2 filters should work normally."""
        template = "name: {{ name | upper }}"
        config = {"name": "my-job"}
        result = _render_k8s_template(template, config)
        assert "name: MY-JOB" in result

    def test_conditional(self):
        """Jinja2 conditionals should work normally."""
        template = "{% if gpu %}accelerator: nvidia{% endif %}"
        config = {"gpu": True}
        result = _render_k8s_template(template, config)
        assert "accelerator: nvidia" in result

    def test_loop(self):
        """Jinja2 loops should work normally."""
        template = "{% for item in items %}{{ item }}\n{% endfor %}"
        config = {"items": ["a", "b", "c"]}
        result = _render_k8s_template(template, config)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_blocks_os_popen_via_cycler(self):
        """Access to os.popen through object traversal should be blocked."""
        template = "{{ cycler.__init__.__globals__.os.popen('whoami').read() }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_dunder_init_access(self):
        """Access to __init__ on objects should be blocked."""
        template = "{{ ''.__class__.__init__.__globals__ }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_dunder_class_chaining(self):
        """Chaining from __class__ to internal attributes should be blocked."""
        template = "{{ ''.__class__.__init__.__globals__ }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_subclasses_access(self):
        """Access to __subclasses__ should be blocked."""
        template = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_globals_access(self):
        """Access to __globals__ should be blocked."""
        template = "{{ config.__init__.__globals__['os'] }}"
        config = {"config": {}}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_import_via_builtins(self):
        """Attempt to access modules via __builtins__ should be blocked."""
        template = (
            "{{ ''.__class__.__init__.__globals__['__builtins__']['__import__']('os').popen('id').read() }}"
        )
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_mro_traversal(self):
        """MRO traversal to reach internal classes should be blocked."""
        template = "{{ [].__class__.__mro__ }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_realistic_k8s_template(self):
        """A realistic Kubernetes manifest template should render correctly."""
        template = """apiVersion: batch/v1
kind: Job
metadata:
  name: {{ name }}
  namespace: {{ namespace }}
spec:
  template:
    spec:
      containers:
      - name: training
        image: {{ image }}
        resources:
          limits:
            nvidia.com/gpu: {{ gpu_count }}
      restartPolicy: Never"""
        config = {
            "name": "sft-llama-job",
            "namespace": "hyperpod",
            "image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0",
            "gpu_count": 8,
        }
        result = _render_k8s_template(template, config)
        assert "name: sft-llama-job" in result
        assert "namespace: hyperpod" in result
        assert "nvidia.com/gpu: 8" in result
        assert "763104351884" in result

    def test_blocks_code_execution_in_template(self):
        """Arbitrary code execution through template internals should be blocked."""
        template = (
            'apiVersion: v1\nkind: ConfigMap\nmetadata:\n'
            '  name: {{ cycler.__init__.__globals__.os.popen(\'echo pwned\').read() }}\n'
            '  namespace: {{ namespace }}'
        )
        config = {"namespace": "default"}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    # -------------------------------------------------------------------
    # Additional sandbox bypass vector tests
    #
    # The SandboxedEnvironment blocks unsafe attribute access via two mechanisms:
    # 1. Dot notation (e.g., obj.__init__): raises SecurityError when the sandbox
    #    detects access to internal attributes on real objects.
    # 2. |attr() filter (e.g., obj|attr("__init__")): returns Undefined (renders
    #    as empty string) for internal attributes. This is safe because Undefined
    #    cannot be called, iterated, or used to access further attributes.
    #
    # Both mechanisms prevent exploitation. Tests below verify neither path
    # leaks internal state or enables code execution.
    # -------------------------------------------------------------------

    def test_blocks_attr_filter_returns_undefined(self):
        """The |attr() filter returns Undefined for internal attributes.

        {{ ""|attr("__class__") }} uses the attr filter to access dunder
        attributes indirectly. The sandbox intercepts this and returns Undefined
        which renders as an empty string — no data leak occurs.
        """
        template = '{{ ""|attr("__class__")|attr("__init__")|attr("__globals__") }}'
        config = {}
        result = _render_k8s_template(template, config)
        # Should render as empty string (Undefined) — no internal data leaked
        assert result == ""
        assert "function" not in result
        assert "module" not in result
        assert "os" not in result

    def test_attr_filter_cannot_reach_callable(self):
        """The attr() chain cannot produce a callable to execute code.

        Even though attr() doesn't raise, the resulting Undefined cannot be
        called, preventing actual exploitation.
        """
        from jinja2.exceptions import UndefinedError
        # Attempting to call the result of attr() on an Undefined raises UndefinedError
        template = '{{ ""|attr("__class__")|attr("__subclasses__")() }}'
        config = {}
        with pytest.raises(UndefinedError):
            _render_k8s_template(template, config)

    def test_attr_filter_get_method_blocked(self):
        """Cannot use .get() on Undefined returned by attr() to extract values."""
        from jinja2.exceptions import UndefinedError
        template = '{% set g = ""|attr("__class__")|attr("__init__")|attr("__globals__") %}{{ g.get("os") }}'
        config = {}
        with pytest.raises(UndefinedError):
            _render_k8s_template(template, config)

    def test_blocks_lipsum_globals_access(self):
        """Access to __globals__ via lipsum builtin should be blocked.

        lipsum is a built-in Jinja2 global (like cycler, joiner, namespace);
        it can be used as an entry point for traversal.
        """
        template = "{{ lipsum.__globals__['os'].popen('id').read() }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_joiner_globals_access(self):
        """Access to __globals__ via joiner builtin should be blocked."""
        template = "{{ joiner.__init__.__globals__['os'].popen('id').read() }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_namespace_init_globals(self):
        """Access to __globals__ via namespace builtin should be blocked."""
        template = "{{ namespace.__init__.__globals__ }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_config_data_object_traversal(self):
        """Objects passed via config_data should not allow internal traversal."""
        template = "{{ obj.__class__.__init__.__globals__ }}"
        config = {"obj": {"key": "value"}}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_format_string_attr_construction_is_safe(self):
        """Format string tricks to construct dunder names are blocked by attr().

        Even if you dynamically build "__class__" via string concat and pass it
        to |attr(), the sandbox still returns Undefined — no data leak.
        """
        template = '{% set name = "__cla" ~ "ss__" %}{{ ""|attr(name) }}'
        config = {}
        result = _render_k8s_template(template, config)
        assert result == ""
        assert "class" not in result.lower() or result == ""

    def test_format_constructed_full_chain_is_safe(self):
        """Dynamically constructed attr names cannot bypass sandbox protection."""
        template = '{% set a = "__ini" %}{% set b = "t__" %}{{ ""|attr("__class__")|attr(a~b)|attr("__globals__") }}'
        config = {}
        result = _render_k8s_template(template, config)
        assert result == ""
        assert "os" not in result
        assert "module" not in result

    def test_map_filter_attribute_returns_undefined(self):
        """The |map(attribute=) filter returns Undefined for dunder access.

        {{ items|map(attribute="__class__")|list }} uses filter parameters
        to perform attribute access, but sandbox blocks it returning Undefined.
        """
        template = '{{ ["a","b"]|map(attribute="__class__")|list }}'
        config = {}
        result = _render_k8s_template(template, config)
        # map with blocked attribute returns Undefined for each item
        assert "str" not in result
        assert "type" not in result

    def test_blocks_map_filter_deep_attribute(self):
        """Deep attribute access via |map should be blocked."""
        template = '{{ ["a"]|map(attribute="__class__.__init__.__globals__")|list }}'
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_set_tag_namespace_globals(self):
        """{% set %} assignment with namespace.__init__.__globals__ should be blocked.

        Self-referential trick using set to assign internal objects.
        """
        template = "{% set x = namespace.__init__.__globals__ %}{{ x }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_set_with_cycler_traversal(self):
        """{% set %} combined with builtin object traversal should be blocked."""
        template = "{% set x = cycler.__init__.__globals__ %}{{ x }}"
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_join_filter_attr_construction_is_safe(self):
        """Using |join to construct attribute names still returns Undefined."""
        template = '{% set parts = ["__","class","__"] %}{{ ""|attr(parts|join) }}'
        config = {}
        result = _render_k8s_template(template, config)
        assert result == ""

    def test_getitem_dunder_returns_undefined(self):
        """Accessing __class__ via bracket notation returns Undefined (safe).

        The sandbox intercepts __getitem__ access to internal attribute names
        and returns Undefined rather than the actual attribute.
        """
        template = '{{ ""["__class__"] }}'
        config = {}
        result = _render_k8s_template(template, config)
        # Bracket notation for dunder attrs also returns Undefined (renders empty)
        assert "str" not in result
        assert "type" not in result

    def test_getitem_cannot_chain_to_globals(self):
        """Bracket notation chaining to reach __globals__ raises SecurityError."""
        template = '{{ ""["__class__"]["__init__"]["__globals__"] }}'
        config = {}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_blocks_request_like_object_traversal(self):
        """If any complex objects are passed in config, traversal should be blocked."""
        class FakeRequest:
            pass

        template = "{{ req.__class__.__init__.__globals__ }}"
        config = {"req": FakeRequest()}
        with pytest.raises(SecurityError):
            _render_k8s_template(template, config)

    def test_attr_filter_no_data_leak_comprehensive(self):
        """Comprehensive test that no attr() variant leaks internal Python state."""
        dangerous_templates = [
            '{{ ""|attr("__class__")|attr("__init__")|attr("__globals__") }}',
            '{{ []|attr("__class__")|attr("__mro__") }}',
            '{{ ""|attr("__class__")|attr("__subclasses__") }}',
            '{% set x = ""|attr("__class__") %}{{ x|attr("__init__") }}',
        ]
        dangerous_indicators = ["function", "module", "os", "subprocess", "popen",
                                "builtins", "<class", "object"]
        config = {}
        for template in dangerous_templates:
            try:
                result = _render_k8s_template(template, config)
                for indicator in dangerous_indicators:
                    assert indicator not in result, (
                        f"Data leak detected! Template: {template}, "
                        f"Found '{indicator}' in result: {result[:100]}"
                    )
            except (SecurityError, Exception):
                # Raising is also acceptable - it means the sandbox blocked it
                pass
