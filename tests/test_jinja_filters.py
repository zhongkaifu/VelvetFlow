import re

from velvetflow.jinja_utils import render_jinja_template


def test_date_filter_formats_now():
    rendered = render_jinja_template("{{ 'now' | date('yyyy-MM-dd') }}", {})
    assert re.match(r"\d{4}-\d{2}-\d{2}", rendered)
