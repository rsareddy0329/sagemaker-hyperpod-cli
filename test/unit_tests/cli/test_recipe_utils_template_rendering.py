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
