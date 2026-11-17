import nbformat as nbf
import yaml
from glob import glob

# Page descriptions for search results and link previews
with open("descriptions.yml") as f:
    DESCRIPTIONS = yaml.safe_load(f)

# Text to look for in adding tags
text_search_dict = {
    "# Solution": "remove-cell",
}


def add_tags(ntbk):
    """Add tags to cells that contain the search text."""
    for cell in ntbk.cells:
        cell_tags = []
        for key, val in text_search_dict.items():
            if key in cell["source"]:
                if val not in cell_tags:
                    cell_tags.append(val)
        if len(cell_tags) > 0:
            cell["metadata"]["tags"] = cell_tags


def add_description(ntbk, path):
    """Write the page description into the notebook metadata.

    myst-nb passes notebook metadata through as front matter: html_meta becomes
    <meta name="description">; og:description overrides the link-preview text,
    which would otherwise be the opening paragraph.
    """
    desc = DESCRIPTIONS.get(path.removesuffix(".ipynb"))
    if not desc:
        print("no description for", path)
        return
    ntbk.metadata["myst"] = {"html_meta": {"description": desc}}
    ntbk.metadata["og:description"] = desc


# Collect a list of all notebooks in the content folder
for path in sorted(glob("*.ipynb")):
    ntbk = nbf.read(path, nbf.NO_CONVERT)
    add_tags(ntbk)
    add_description(ntbk, path)
    nbf.write(ntbk, path)
