from lxml import etree

def parse_cvat_xml(xml_path: str):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    image_el = root.find(".//image")
    if image_el is None:
        raise ValueError("<image> not found in XML")

    meta = {
        "name":   image_el.get("name"),
        "width":  int(image_el.get("width") or 0),
        "height": int(image_el.get("height") or 0),
    }

    rep_points: dict[str, tuple[float, float]] = {}
    polygons:   list[dict[str, Any]]           = []
    label_counter: dict[str, int]              = {}

    for child in image_el.iter():
        if child is image_el:
            continue
        tag = child.tag
        if not isinstance(tag, str):
            continue
        tag = tag.lower()

        label   = (child.get("label") or "").strip()
        pts_raw = (child.get("points") or "").strip()
        if not pts_raw:
            continue

        if tag == "points" and label.startswith("rep_point"):
            x, y = pts_raw.split(";")[0].split(",")
            rep_points[label] = (float(x), float(y))

        elif tag == "polygon" and label:
            if label.lower() == "work_area":
                continue
            coords = []
            for pair in pts_raw.split(";"):
                x, y = pair.split(",")
                coords.append((float(x), float(y)))

            label_counter[label] = label_counter.get(label, 0) + 1
            unique_label = f"{label}_{label_counter[label]}"
            polygons.append({"label": unique_label, "points": coords})

    return rep_points, polygons, meta
