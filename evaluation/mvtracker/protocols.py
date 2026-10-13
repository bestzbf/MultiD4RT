"""Released MVTracker test configurations (never sample replacement queries)."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrackingProtocol:
    name: str
    official_name: str
    family: str
    folder: str
    views: tuple[int, ...]
    sequences: int
    frames: int | None
    queries: int
    cache_name: str
    depth: str

    def to_dict(self):
        result = asdict(self)
        result["views"] = list(self.views)
        return result


def panoptic(name, views):
    view_string = "_".join(map(str, views))
    return TrackingProtocol(name, f"panoptic-multiview-views{view_string}-cached",
        "panoptic-multiview", "panoptic-multiview", views, 6, 150, 384,
        f"cachedtracks--seed72-views{view_string}-n384--v1.npz", "dynamic3dgs")


def dex(name, estimated=False):
    return TrackingProtocol(name,
        "dex-ycb-multiview-duster0123-cached" if estimated else "dex-ycb-multiview-views0123-cached",
        "dexycb-multiview", "dex-ycb-multiview", (0, 1, 2, 3), 10, None, 384,
        "cachedtracks--seed72-views0_1_2_3-n384--v1.npz", "dust3r" if estimated else "sensor")


def kubric(name, estimated=False):
    return TrackingProtocol(name,
        "kubric-multiview-v3-duster0123-cached" if estimated else "kubric-multiview-v3-views0123-cached",
        "kubric-multiview", "kubric-multiview/test", (0, 1, 2, 3), 30, 24, 512,
        "cachedtracks--seed72-dynamic0.5-verydynamic-0.25-views0_1_2_3-n512-numviews-1-t24--v1.npz",
        "dust3r" if estimated else "rendered")


PROTOCOLS = {p.name: p for p in (
    panoptic("panoptic", (1, 7, 14, 20)),
    panoptic("panoptic-27_16_14_8", (27, 16, 14, 8)),
    panoptic("panoptic-1_4_7_11", (1, 4, 7, 11)),
    dex("dexycb"), dex("dexycb-dust3r", True),
    kubric("mv-kubric"), kubric("mv-kubric-dust3r", True),
)}
DEFAULT_SUITE = ("panoptic", "dexycb", "mv-kubric")
N3DV_SCENES = ("coffee_martini", "cook_spinach", "cut_roasted_beef",
               "flame_salmon_1", "flame_steak", "sear_steak")
