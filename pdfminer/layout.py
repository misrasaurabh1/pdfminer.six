import heapq
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import (
    Generic,
    TypeVar,
    Union,
)

from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfexceptions import PDFTypeError, PDFValueError
from pdfminer.pdffont import PDFFont
from pdfminer.pdfinterp import Color, PDFGraphicState
from pdfminer.pdftypes import PDFStream
from pdfminer.utils import (
    INF,
    LTComponentT,
    Matrix,
    PathSegment,
    Plane,
    Point,
    Rect,
    apply_matrix_rect,
    bbox2str,
    fsplit,
    get_bound,
    matrix2str,
    uniq,
)

logger = logging.getLogger(__name__)


try:
    from pdfminer_core import (  # type: ignore[import]
        group_chars_into_lines as _group_chars_into_lines,
        group_textlines_fast as _group_textlines_fast,
        compute_textbox_distances as _compute_textbox_distances,
        compute_word_gaps as _compute_word_gaps,
        expand_bbox as _expand_bbox,
        compute_group_bbox as _compute_group_bbox,
    )
    from pdfminer.utils import _RustPlane as _FastPlane
    _HAS_RUST = True
    _PlaneClass: type = _FastPlane
except ImportError:
    _HAS_RUST = False
    _PlaneClass = Plane
    _expand_bbox = None


class IndexAssigner:
    def __init__(self, index: int = 0) -> None:
        self.index = index

    def run(self, obj: "LTItem") -> None:
        if isinstance(obj, LTTextBox):
            obj.index = self.index
            self.index += 1
        elif isinstance(obj, LTTextGroup):
            for x in obj:
                self.run(x)


class LAParams:
    """Parameters for layout analysis

    :param line_overlap: If two characters have more overlap than this they
        are considered to be on the same line. The overlap is specified
        relative to the minimum height of both characters.
    :param char_margin: If two characters are closer together than this
        margin they are considered part of the same line. The margin is
        specified relative to the width of the character.
    :param word_margin: If two characters on the same line are further apart
        than this margin then they are considered to be two separate words, and
        an intermediate space will be added for readability. The margin is
        specified relative to the width of the character.
    :param line_margin: If two lines are are close together they are
        considered to be part of the same paragraph. The margin is
        specified relative to the height of a line.
    :param boxes_flow: Specifies how much a horizontal and vertical position
        of a text matters when determining the order of text boxes. The value
        should be within the range of -1.0 (only horizontal position
        matters) to +1.0 (only vertical position matters). You can also pass
        `None` to disable advanced layout analysis, and instead return text
        based on the position of the bottom left corner of the text box.
    :param detect_vertical: If vertical text should be considered during
        layout analysis
    :param all_texts: If layout analysis should be performed on text in
        figures.
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: float | None = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
    ) -> None:
        self.line_overlap = line_overlap
        self.char_margin = char_margin
        self.line_margin = line_margin
        self.word_margin = word_margin
        self.boxes_flow = boxes_flow
        self.detect_vertical = detect_vertical
        self.all_texts = all_texts

        self._validate()

    def _validate(self) -> None:
        if self.boxes_flow is not None:
            boxes_flow_err_msg = (
                "LAParam boxes_flow should be None, or a number between -1 and +1"
            )
            if not (isinstance(self.boxes_flow, (int, float))):
                raise PDFTypeError(boxes_flow_err_msg)
            if not -1 <= self.boxes_flow <= 1:
                raise PDFValueError(boxes_flow_err_msg)

    def __repr__(self) -> str:
        return (
            f"<LAParams: char_margin={self.char_margin:.1f}, "
            f"line_margin={self.line_margin:.1f}, "
            f"word_margin={self.word_margin:.1f} "
            f"all_texts={self.all_texts!r}>"
        )


class LTItem:
    """Interface for things that can be analyzed"""

    __slots__ = ()

    def analyze(self, laparams: LAParams) -> None:
        """Perform the layout analysis."""


class LTText:
    """Interface for things that have text"""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.get_text()!r}>"

    def get_text(self) -> str:
        """Text contained in this object"""
        raise NotImplementedError


class LTComponent(LTItem):
    """Object with a bounding box"""

    __slots__ = ("bbox", "x0", "y0", "x1", "y1", "width", "height")

    # Integer type tag for fast dispatch (class-level constant, no per-instance cost).
    # 0 = generic LTComponent / unknown subclass
    # 1 = LTChar
    # 3 = LTTextLine (and subclasses)
    # 4 = LTTextBox (and subclasses)
    _type_tag: int = 0

    def __init__(self, bbox: Rect) -> None:
        LTItem.__init__(self)
        self.set_bbox(bbox)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {bbox2str(self.bbox)}>"

    # Disable comparison.
    def __lt__(self, _: object) -> bool:
        raise PDFValueError

    def __le__(self, _: object) -> bool:
        raise PDFValueError

    def __gt__(self, _: object) -> bool:
        raise PDFValueError

    def __ge__(self, _: object) -> bool:
        raise PDFValueError

    def set_bbox(self, bbox: Rect) -> None:
        (x0, y0, x1, y1) = bbox
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0
        self.bbox = bbox

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def is_hoverlap(self, obj: "LTComponent") -> bool:
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.x0 <= self.x1 and self.x0 <= obj.x1

    def hdistance(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return 0
        else:
            return min(abs(self.x0 - obj.x1), abs(self.x1 - obj.x0))

    def hoverlap(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return min(abs(self.x0 - obj.x1), abs(self.x1 - obj.x0))
        else:
            return 0

    def is_voverlap(self, obj: "LTComponent") -> bool:
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.y0 <= self.y1 and self.y0 <= obj.y1

    def vdistance(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return 0
        else:
            return min(abs(self.y0 - obj.y1), abs(self.y1 - obj.y0))

    def voverlap(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return min(abs(self.y0 - obj.y1), abs(self.y1 - obj.y0))
        else:
            return 0


class LTCurve(LTComponent):
    """A generic Bezier curve

    The parameter `original_path` contains the original
    pathing information from the pdf (e.g. for reconstructing Bezier Curves).

    `dashing_style` contains the Dashing information if any.
    """
    __slots__ = (
        "pts", "linewidth", "stroke", "fill", "evenodd",
        "stroking_color", "non_stroking_color", "original_path", "dashing_style",
    )


    def __init__(
        self,
        linewidth: float,
        pts: list[Point],
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Color | None = None,
        non_stroking_color: Color | None = None,
        original_path: list[PathSegment] | None = None,
        dashing_style: tuple[object, object] | None = None,
    ) -> None:
        LTComponent.__init__(self, get_bound(pts))
        self.pts = pts
        self.linewidth = linewidth
        self.stroke = stroke
        self.fill = fill
        self.evenodd = evenodd
        self.stroking_color = stroking_color
        self.non_stroking_color = non_stroking_color
        self.original_path = original_path
        self.dashing_style = dashing_style

    def get_pts(self) -> str:
        return ",".join("{:.3f},{:.3f}".format(*p) for p in self.pts)


class LTLine(LTCurve):
    """A single straight line.

    Could be used for separating text or figures.
    """
    __slots__ = ()


    def __init__(
        self,
        linewidth: float,
        p0: Point,
        p1: Point,
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Color | None = None,
        non_stroking_color: Color | None = None,
        original_path: list[PathSegment] | None = None,
        dashing_style: tuple[object, object] | None = None,
    ) -> None:
        LTCurve.__init__(
            self,
            linewidth,
            [p0, p1],
            stroke,
            fill,
            evenodd,
            stroking_color,
            non_stroking_color,
            original_path,
            dashing_style,
        )


class LTRect(LTCurve):
    """A rectangle.

    Could be used for framing another pictures or figures.
    """
    __slots__ = ()


    def __init__(
        self,
        linewidth: float,
        bbox: Rect,
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Color | None = None,
        non_stroking_color: Color | None = None,
        original_path: list[PathSegment] | None = None,
        dashing_style: tuple[object, object] | None = None,
    ) -> None:
        (x0, y0, x1, y1) = bbox
        LTCurve.__init__(
            self,
            linewidth,
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            stroke,
            fill,
            evenodd,
            stroking_color,
            non_stroking_color,
            original_path,
            dashing_style,
        )


class LTImage(LTComponent):
    """An image object.

    Embedded images can be in JPEG, Bitmap or JBIG2.
    """
    __slots__ = ("name", "stream", "srcsize", "imagemask", "bits", "colorspace")


    def __init__(self, name: str, stream: PDFStream, bbox: Rect) -> None:
        LTComponent.__init__(self, bbox)
        self.name = name
        self.stream = stream
        self.srcsize = (stream.get_any(("W", "Width")), stream.get_any(("H", "Height")))
        self.imagemask = stream.get_any(("IM", "ImageMask"))
        self.bits = stream.get_any(("BPC", "BitsPerComponent"), 1)
        self.colorspace = stream.get_any(("CS", "ColorSpace"))
        if not isinstance(self.colorspace, list):
            self.colorspace = [self.colorspace]

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}({self.name}) "
            f"{bbox2str(self.bbox)} {self.srcsize!r}>"
        )


class LTAnno(LTItem, LTText):
    __slots__ = ("_text",)

    """Actual letter in the text as a Unicode string.

    Note that, while a LTChar object has actual boundaries, LTAnno objects does
    not, as these are "virtual" characters, inserted by a layout analyzer
    according to the relationship between two characters (e.g. a space).
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class LTChar(LTComponent, LTText):
    """Actual letter in the text as a Unicode string."""
    __slots__ = (
        "matrix", "_text", "fontname", "ncs", "graphicstate",
        "size", "adv", "upright", "rendermode",
    )


    _type_tag: int = 1

    def __init__(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: float | tuple[float | None, float],
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
    ) -> None:
        # Bypass LTText.__init__ (no-op) and LTComponent.__init__ (calls set_bbox)
        # to eliminate 2 extra call frames per character (125k chars/run).
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.ncs = ncs
        self.graphicstate = graphicstate
        self.adv = textwidth * fontsize * scaling
        vertical = font.is_vertical()
        if vertical:
            assert isinstance(textdisp, tuple)
            (vx, vy) = textdisp
            vx = fontsize * 0.5 if vx is None else vx * fontsize * 0.001
            vy = (1000 - vy) * fontsize * 0.001
            bbox = (-vx, vy + rise + self.adv, -vx + fontsize, vy + rise)
        else:
            descent = font.get_descent() * fontsize
            bbox = (0, descent + rise, self.adv, descent + rise + fontsize)
        (a, b, c, d, _e, _f) = self.matrix
        self.upright = a * d * scaling > 0 and b * c <= 0
        (x0, y0, x1, y1) = apply_matrix_rect(self.matrix, bbox)
        if x1 < x0:
            (x0, x1) = (x1, x0)
        if y1 < y0:
            (y0, y1) = (y1, y0)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0
        self.bbox = (x0, y0, x1, y1)
        self.rendermode = 0
        self.size = self.width if vertical else self.height

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {bbox2str(self.bbox)} "
            f"matrix={matrix2str(self.matrix)} "
            f"font={self.fontname!r} "
            f"adv={self.adv} "
            f"text={self.get_text()!r}>"
        )

    def get_text(self) -> str:
        return self._text


LTItemT = TypeVar("LTItemT", bound=LTItem)


class LTContainer(LTComponent, Generic[LTItemT]):
    """Object that can be extended and analyzed"""
    __slots__ = ("_objs",)


    def __init__(self, bbox: Rect) -> None:
        LTComponent.__init__(self, bbox)
        self._objs: list[LTItemT] = []

    def __iter__(self) -> Iterator[LTItemT]:
        return iter(self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def add(self, obj: LTItemT) -> None:
        self._objs.append(obj)

    def extend(self, objs: Iterable[LTItemT]) -> None:
        for obj in objs:
            self.add(obj)

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)


class LTExpandableContainer(LTContainer[LTItemT]):
    def __init__(self) -> None:
        LTContainer.__init__(self, (+INF, +INF, -INF, -INF))

    # Incompatible override: we take an LTComponent (with bounding box), but
    # super() LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        LTContainer.add(self, obj)  # type: ignore[arg-type]
        if _expand_bbox is not None:
            new_bbox = _expand_bbox(self.bbox, (obj.x0, obj.y0, obj.x1, obj.y1))
        else:
            new_bbox = (
                min(self.x0, obj.x0),
                min(self.y0, obj.y0),
                max(self.x1, obj.x1),
                max(self.y1, obj.y1),
            )
        self.set_bbox(new_bbox)


class LTTextContainer(LTExpandableContainer[LTItemT], LTText):
    __slots__ = ()

    def __init__(self) -> None:
        LTText.__init__(self)
        LTExpandableContainer.__init__(self)

    def get_text(self) -> str:
        return "".join(
            obj.get_text() for obj in self if isinstance(obj, LTText)  # type: ignore[union-attr]
        )


TextLineElement = Union[LTChar, LTAnno]


class LTTextLine(LTTextContainer[TextLineElement]):
    """Contains a list of LTChar objects that represent a single text line.

    The characters are aligned either horizontally or vertically, depending on
    the text's writing mode.
    """
    __slots__ = ("word_margin",)


    _type_tag: int = 3
    # Subclasses override _is_horizontal; default True for LTTextLine itself.
    _is_horizontal: bool = True

    def __init__(self, word_margin: float) -> None:
        super().__init__()
        self.word_margin = word_margin

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {bbox2str(self.bbox)} {self.get_text()!r}>"

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)
        LTContainer.add(self, LTAnno("\n"))

    def find_neighbors(
        self,
        plane: Plane[LTComponentT],
        ratio: float,
    ) -> list["LTTextLine"]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        if super().is_empty():
            return True
        return all(obj.get_text().isspace() for obj in self._objs)


class LTTextLineHorizontal(LTTextLine):
    __slots__ = ("_x1",)

    # _is_horizontal inherited as True from LTTextLine

    def __init__(self, word_margin: float) -> None:
        LTTextLine.__init__(self, word_margin)
        self._x1: float = +INF

    # Incompatible override: we take an LTComponent (with bounding box), but
    # LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        if obj._type_tag == 1 and self.word_margin:  # LTChar._type_tag == 1
            margin = self.word_margin * max(obj.width, obj.height)
            if self._x1 < obj.x0 - margin:
                LTContainer.add(self, LTAnno(" "))
        self._x1 = obj.x1
        super().add(obj)

    def find_neighbors(
        self,
        plane: Plane[LTComponentT],
        ratio: float,
    ) -> list[LTTextLine]:
        """Finds neighboring LTTextLineHorizontals in the plane.

        Returns a list of other LTTestLineHorizontals in the plane which are
        close to self. "Close" can be controlled by ratio. The returned objects
        will be the same height as self, and also either left-, right-, or
        centrally-aligned.
        """
        d = ratio * self.height
        objs = plane.find((self.x0, self.y0 - d, self.x1, self.y1 + d))
        return [
            obj
            for obj in objs
            if (
                isinstance(obj, LTTextLineHorizontal)
                and self._is_same_height_as(obj, tolerance=d)
                and (
                    self._is_left_aligned_with(obj, tolerance=d)
                    or self._is_right_aligned_with(obj, tolerance=d)
                    or self._is_centrally_aligned_with(obj, tolerance=d)
                )
            )
        ]

    def _is_left_aligned_with(self, other: LTComponent, tolerance: float = 0) -> bool:
        """Whether the left-hand edge of `other` is within `tolerance`."""
        return abs(other.x0 - self.x0) <= tolerance

    def _is_right_aligned_with(self, other: LTComponent, tolerance: float = 0) -> bool:
        """Whether the right-hand edge of `other` is within `tolerance`."""
        return abs(other.x1 - self.x1) <= tolerance

    def _is_centrally_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0,
    ) -> bool:
        """Whether the horizontal center of `other` is within `tolerance`."""
        return abs((other.x0 + other.x1) / 2 - (self.x0 + self.x1) / 2) <= tolerance

    def _is_same_height_as(self, other: LTComponent, tolerance: float = 0) -> bool:
        return abs(other.height - self.height) <= tolerance


class LTTextLineVertical(LTTextLine):
    __slots__ = ("_y0",)

    _is_horizontal: bool = False

    def __init__(self, word_margin: float) -> None:
        LTTextLine.__init__(self, word_margin)
        self._y0: float = -INF

    # Incompatible override: we take an LTComponent (with bounding box), but
    # LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        if obj._type_tag == 1 and self.word_margin:  # LTChar._type_tag == 1
            margin = self.word_margin * max(obj.width, obj.height)
            if obj.y1 + margin < self._y0:
                LTContainer.add(self, LTAnno(" "))
        self._y0 = obj.y0
        super().add(obj)

    def find_neighbors(
        self,
        plane: Plane[LTComponentT],
        ratio: float,
    ) -> list[LTTextLine]:
        """Finds neighboring LTTextLineVerticals in the plane.

        Returns a list of other LTTextLineVerticals in the plane which are
        close to self. "Close" can be controlled by ratio. The returned objects
        will be the same width as self, and also either upper-, lower-, or
        centrally-aligned.
        """
        d = ratio * self.width
        objs = plane.find((self.x0 - d, self.y0, self.x1 + d, self.y1))
        return [
            obj
            for obj in objs
            if (
                isinstance(obj, LTTextLineVertical)
                and self._is_same_width_as(obj, tolerance=d)
                and (
                    self._is_lower_aligned_with(obj, tolerance=d)
                    or self._is_upper_aligned_with(obj, tolerance=d)
                    or self._is_centrally_aligned_with(obj, tolerance=d)
                )
            )
        ]

    def _is_lower_aligned_with(self, other: LTComponent, tolerance: float = 0) -> bool:
        """Whether the lower edge of `other` is within `tolerance`."""
        return abs(other.y0 - self.y0) <= tolerance

    def _is_upper_aligned_with(self, other: LTComponent, tolerance: float = 0) -> bool:
        """Whether the upper edge of `other` is within `tolerance`."""
        return abs(other.y1 - self.y1) <= tolerance

    def _is_centrally_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0,
    ) -> bool:
        """Whether the vertical center of `other` is within `tolerance`."""
        return abs((other.y0 + other.y1) / 2 - (self.y0 + self.y1) / 2) <= tolerance

    def _is_same_width_as(self, other: LTComponent, tolerance: float) -> bool:
        return abs(other.width - self.width) <= tolerance


class LTTextBox(LTTextContainer[LTTextLine]):
    """Represents a group of text chunks in a rectangular area.

    Note that this box is created by geometric analysis and does not
    necessarily represents a logical boundary of the text. It contains a list
    of LTTextLine objects.
    """
    __slots__ = ("index",)

    _type_tag: int = 4
    # True for horizontal (LR-TB) boxes; LTTextBoxVertical overrides to False.
    _is_horizontal: bool = True

    def __init__(self) -> None:
        LTTextContainer.__init__(self)
        self.index: int = -1

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}({self.index}) "
            f"{bbox2str(self.bbox)} {self.get_text()!r}>"
        )

    def get_writing_mode(self) -> str:
        raise NotImplementedError


class LTTextBoxHorizontal(LTTextBox):
    __slots__ = ()

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        self._objs.sort(key=lambda obj: -obj.y1)

    def get_writing_mode(self) -> str:
        return "lr-tb"


class LTTextBoxVertical(LTTextBox):
    __slots__ = ()

    _is_horizontal: bool = False

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        self._objs.sort(key=lambda obj: -obj.x1)

    def get_writing_mode(self) -> str:
        return "tb-rl"


TextGroupElement = Union[LTTextBox, "LTTextGroup"]


class LTTextGroup(LTTextContainer[TextGroupElement]):
    __slots__ = ()

    # True for LR-TB groups; LTTextGroupTBRL overrides to False.
    _is_horizontal: bool = True

    def __init__(self, objs: Iterable[TextGroupElement]) -> None:
        super().__init__()
        self.extend(objs)


class LTTextGroupLRTB(LTTextGroup):
    __slots__ = ()

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        assert laparams.boxes_flow is not None
        boxes_flow = laparams.boxes_flow
        # reorder the objects from top-left to bottom-right.
        self._objs.sort(
            key=lambda obj: (1 - boxes_flow) * obj.x0
            - (1 + boxes_flow) * (obj.y0 + obj.y1),
        )


class LTTextGroupTBRL(LTTextGroup):
    __slots__ = ()

    _is_horizontal: bool = False

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        assert laparams.boxes_flow is not None
        boxes_flow = laparams.boxes_flow
        # reorder the objects from top-right to bottom-left.
        self._objs.sort(
            key=lambda obj: -(1 + boxes_flow) * (obj.x0 + obj.x1)
            - (1 - boxes_flow) * obj.y1,
        )


class LTLayoutContainer(LTContainer[LTComponent]):
    __slots__ = ("groups",)

    def __init__(self, bbox: Rect) -> None:
        LTContainer.__init__(self, bbox)
        self.groups: list[LTTextGroup] | None = None

    def add_chars_batch(self, chars: "list[LTChar]") -> None:
        """Add multiple LTChar objects, updating the container bbox only once.

        This is faster than calling add() 125k times per page because it
        avoids expanding the bbox on every character insertion.
        """
        if not chars:
            return
        self._objs.extend(chars)
        bx0 = bx1 = chars[0].x0
        by0 = by1 = chars[0].y0
        for c in chars:
            if c.x0 < bx0:
                bx0 = c.x0
            if c.y0 < by0:
                by0 = c.y0
            if c.x1 > bx1:
                bx1 = c.x1
            if c.y1 > by1:
                by1 = c.y1
        self.set_bbox(
            (
                min(self.x0, bx0),
                min(self.y0, by0),
                max(self.x1, bx1),
                max(self.y1, by1),
            )
        )

    # group_objects: group text object to textlines.
    def group_objects(
        self,
        laparams: LAParams,
        objs: Iterable[LTComponent],
    ) -> Iterator[LTTextLine]:
        obj0 = None
        line: LTTextLine | None = None
        for obj1 in objs:
            if obj0 is not None:
                # halign: obj0 and obj1 is horizontally aligned.
                #
                #   +------+ - - -
                #   | obj0 | - - +------+   -
                #   |      |     | obj1 |   | (line_overlap)
                #   +------+ - - |      |   -
                #          - - - +------+
                #
                #          |<--->|
                #        (char_margin)
                halign = (
                    obj0.is_voverlap(obj1)
                    and min(obj0.height, obj1.height) * laparams.line_overlap
                    < obj0.voverlap(obj1)
                    and obj0.hdistance(obj1)
                    < max(obj0.width, obj1.width) * laparams.char_margin
                )

                # valign: obj0 and obj1 is vertically aligned.
                #
                #   +------+
                #   | obj0 |
                #   |      |
                #   +------+ - - -
                #     |    |     | (char_margin)
                #     +------+ - -
                #     | obj1 |
                #     |      |
                #     +------+
                #
                #     |<-->|
                #   (line_overlap)
                valign = (
                    laparams.detect_vertical
                    and obj0.is_hoverlap(obj1)
                    and min(obj0.width, obj1.width) * laparams.line_overlap
                    < obj0.hoverlap(obj1)
                    and obj0.vdistance(obj1)
                    < max(obj0.height, obj1.height) * laparams.char_margin
                )

                if line is not None and (
                    (halign and line._is_horizontal)
                    or (valign and not line._is_horizontal)
                ):
                    line.add(obj1)
                elif line is not None:
                    yield line
                    line = None
                elif valign and not halign:
                    line = LTTextLineVertical(laparams.word_margin)
                    line.add(obj0)
                    line.add(obj1)
                elif halign and not valign:
                    line = LTTextLineHorizontal(laparams.word_margin)
                    line.add(obj0)
                    line.add(obj1)
                else:
                    line = LTTextLineHorizontal(laparams.word_margin)
                    line.add(obj0)
                    yield line
                    line = None
            obj0 = obj1
        if line is None:
            line = LTTextLineHorizontal(laparams.word_margin)
            assert obj0 is not None
            line.add(obj0)
        yield line

    def group_textlines(
        self,
        laparams: LAParams,
        lines: Iterable[LTTextLine],
    ) -> Iterator[LTTextBox]:
        """Group neighboring lines to textboxes"""
        lines = list(lines)
        if _HAS_RUST and lines:
            yield from self._group_textlines_rust(laparams, lines)
            return
        plane: Plane[LTTextLine] = _PlaneClass(self.bbox)
        plane.extend(lines)
        boxes: dict[LTTextLine, LTTextBox] = {}
        for line in lines:
            neighbors = line.find_neighbors(plane, laparams.line_margin)
            members = [line]
            for obj1 in neighbors:
                members.append(obj1)
                if obj1 in boxes:
                    members.extend(boxes.pop(obj1))
            if line._is_horizontal:
                box: LTTextBox = LTTextBoxHorizontal()
            else:
                box = LTTextBoxVertical()
            for obj in uniq(members):
                box.add(obj)
                boxes[obj] = box
        done: set[LTTextBox] = set()
        for line in lines:
            if line not in boxes:
                continue
            box = boxes[line]
            if box in done:
                continue
            done.add(box)
            if not box.is_empty():
                yield box

    def _group_textlines_rust(
        self,
        laparams: LAParams,
        lines: list[LTTextLine],
    ) -> Iterator[LTTextBox]:
        """Rust-accelerated group_textlines using spatial union-find."""
        bboxes = [
            (l.x0, l.y0, l.x1, l.y1, l._is_horizontal)
            for l in lines
        ]
        groups = _group_textlines_fast(
            bboxes,
            laparams.line_margin,
            50.0,
            self.bbox,
        )
        for group_indices in groups:
            first = lines[group_indices[0]]
            box: LTTextBox = (
                LTTextBoxHorizontal()
                if first._is_horizontal
                else LTTextBoxVertical()
            )
            for idx in group_indices:
                box.add(lines[idx])
            if not box.is_empty():
                yield box

    def group_textboxes(
        self,
        laparams: LAParams,
        boxes: Sequence[LTTextBox],
    ) -> list[LTTextGroup]:
        """Group textboxes hierarchically.

        Get pair-wise distances, via dist func defined below, and then merge
        from the closest textbox pair. Once obj1 and obj2 are merged /
        grouped, the resulting group is considered as a new object, and its
        distances to other objects & groups are added to the process queue.

        For performance reason, pair-wise distances and object pair info are
        maintained in a heap of (idx, dist, id(obj1), id(obj2), obj1, obj2)
        tuples. It ensures quick access to the smallest element. Note that
        since comparison operators, e.g., __lt__, are disabled for
        LTComponent, id(obj) has to appear before obj in element tuples.

        :param laparams: LAParams object.
        :param boxes: All textbox objects to be grouped.
        :return: a list that has only one element, the final top level group.
        """
        ElementT = Union[LTTextBox, LTTextGroup]
        plane: Plane[ElementT] = _PlaneClass(self.bbox)

        def dist(obj1: LTComponent, obj2: LTComponent) -> float:
            """A distance function between two TextBoxes.

            Consider the bounding rectangle for obj1 and obj2.
            Return its area less the areas of obj1 and obj2,
            shown as 'www' below. This value may be negative.
                    +------+..........+ (x1, y1)
                    | obj1 |wwwwwwwwww:
                    +------+www+------+
                    :wwwwwwwwww| obj2 |
            (x0, y0) +..........+------+
            """
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            return (
                (x1 - x0) * (y1 - y0)
                - obj1.width * obj1.height
                - obj2.width * obj2.height
            )

        def isany(obj1: ElementT, obj2: ElementT) -> bool:
            """Return True if any plane object other than obj1/obj2 overlaps their bounding rect."""
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            bbox = (x0, y0, x1, y1)
            if _HAS_RUST:
                return plane.has_other_overlapping(bbox, obj1, obj2)  # type: ignore[union-attr]
            for obj in plane.find(bbox):
                if obj is not obj1 and obj is not obj2:
                    return True
            return False

        dists: list[tuple[bool, float, int, int, ElementT, ElementT]] = []
        if _HAS_RUST and len(boxes) > 1:
            raw = _compute_textbox_distances(
                [(b.x0, b.y0, b.x1, b.y1) for b in boxes]
            )
            for d_val, i, j in raw:
                dists.append((False, d_val, id(boxes[i]), id(boxes[j]), boxes[i], boxes[j]))
        else:
            for i in range(len(boxes)):
                box1 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    box2 = boxes[j]
                    dists.append((False, dist(box1, box2), id(box1), id(box2), box1, box2))
        heapq.heapify(dists)

        plane.extend(boxes)
        done = set()
        while len(dists) > 0:
            (skip_isany, d, id1, id2, obj1, obj2) = heapq.heappop(dists)
            # Skip objects that are already merged
            if (id1 not in done) and (id2 not in done):
                if not skip_isany and isany(obj1, obj2):
                    heapq.heappush(dists, (True, d, id1, id2, obj1, obj2))
                    continue
                if not obj1._is_horizontal or not obj2._is_horizontal:
                    group: LTTextGroup = LTTextGroupTBRL([obj1, obj2])
                else:
                    group = LTTextGroupLRTB([obj1, obj2])
                plane.remove(obj1)
                plane.remove(obj2)
                done.update([id1, id2])

                for other in plane:
                    heapq.heappush(
                        dists,
                        (False, dist(group, other), id(group), id(other), group, other),
                    )
                plane.add(group)
        # By now only groups are in the plane
        return list(plane)  # type: ignore[return-value]

    def _group_objects_fast(
        self,
        laparams: LAParams,
        objs: list["LTChar"],
    ) -> "Iterator[LTTextLine]":
        """Rust-accelerated version of group_objects for horizontal-only text.

        Uses pdfminer_core.group_chars_into_lines + compute_word_gaps to
        determine grouping and word-space positions, then constructs
        LTTextLineHorizontal objects directly — bypassing the per-character
        overhead of the individual add() loop.  All LT* objects remain Python
        instances for full compatibility.
        """
        char_bboxes = [(obj.x0, obj.y0, obj.x1, obj.y1) for obj in objs]
        groups = _group_chars_into_lines(
            char_bboxes,
            laparams.line_overlap,
            laparams.char_margin,
        )
        _anno_space = LTAnno(" ")
        word_margin = laparams.word_margin
        for char_indices, space_positions in _compute_word_gaps(
            char_bboxes, groups, word_margin
        ):
            lx0, ly0, lx1, ly1 = _compute_group_bbox(char_bboxes, char_indices)
            if space_positions:
                # Interleave LTAnno spaces at the requested positions.
                sp_iter = iter(space_positions)
                next_sp = next(sp_iter, None)
                line_objs: list[LTItem] = []
                for pos, idx in enumerate(char_indices):
                    if pos == next_sp:
                        line_objs.append(_anno_space)
                        next_sp = next(sp_iter, None)
                    line_objs.append(objs[idx])
            else:
                line_objs = [objs[idx] for idx in char_indices]  # type: ignore[misc]

            # Build LTTextLineHorizontal without calling add() for each object.
            line: LTTextLineHorizontal = LTTextLineHorizontal.__new__(
                LTTextLineHorizontal
            )
            line.word_margin = word_margin
            line._objs = line_objs  # type: ignore[assignment]
            line._x1 = lx1
            line.set_bbox((lx0, ly0, lx1, ly1))
            yield line

    def analyze(self, laparams: LAParams) -> None:
        # textobjs is a list of LTChar objects, i.e.
        # it has all the individual characters in the page.
        # Use _type_tag fast path to avoid isinstance overhead in this hot loop.
        textobjs: list[LTChar] = []
        otherobjs: list[LTComponent] = []
        for _obj in self:
            if _obj._type_tag == 1:  # LTChar._type_tag == 1
                textobjs.append(_obj)  # type: ignore[arg-type]
            else:
                otherobjs.append(_obj)
        for obj in otherobjs:
            obj.analyze(laparams)
        if not textobjs:
            return
        if _HAS_RUST and not laparams.detect_vertical:
            textlines = list(
                self._group_objects_fast(laparams, textobjs)
            )
        else:
            textlines = list(self.group_objects(laparams, textobjs))
        (empties, textlines) = fsplit(lambda obj: obj.is_empty(), textlines)
        for obj in empties:
            obj.analyze(laparams)
        textboxes = list(self.group_textlines(laparams, textlines))
        if laparams.boxes_flow is None:
            for textbox in textboxes:
                textbox.analyze(laparams)

            def getkey(box: LTTextBox) -> tuple[int, float, float]:
                if not box._is_horizontal:
                    return (0, -box.x1, -box.y0)
                else:
                    return (1, -box.y0, box.x0)

            textboxes.sort(key=getkey)
        else:
            self.groups = self.group_textboxes(laparams, textboxes)
            assigner = IndexAssigner()
            for group in self.groups:
                group.analyze(laparams)
                assigner.run(group)
            textboxes.sort(key=lambda box: box.index)
        self._objs = (
            textboxes  # type: ignore[assignment]
            + otherobjs
            + empties  # type: ignore[operator]
        )


class LTFigure(LTLayoutContainer):
    """Represents an area used by PDF Form objects.

    PDF Forms can be used to present figures or pictures by embedding yet
    another PDF document within a page. Note that LTFigure objects can appear
    recursively.
    """
    __slots__ = ("name", "matrix")


    def __init__(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self.name = name
        self.matrix = matrix
        (x, y, w, h) = bbox
        rect = (x, y, x + w, y + h)
        bbox = apply_matrix_rect(matrix, rect)
        LTLayoutContainer.__init__(self, bbox)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}({self.name}) "
            f"{bbox2str(self.bbox)} "
            f"matrix={matrix2str(self.matrix)}>"
        )

    def analyze(self, laparams: LAParams) -> None:
        if not laparams.all_texts:
            return
        LTLayoutContainer.analyze(self, laparams)


class LTPage(LTLayoutContainer):
    """Represents an entire page.

    Like any other LTLayoutContainer, an LTPage can be iterated to obtain child
    objects like LTTextBox, LTFigure, LTImage, LTRect, LTCurve and LTLine.
    """
    __slots__ = ("pageid", "rotate")


    def __init__(self, pageid: int, bbox: Rect, rotate: float = 0) -> None:
        LTLayoutContainer.__init__(self, bbox)
        self.pageid = pageid
        self.rotate = rotate

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}({self.pageid!r}) "
            f"{bbox2str(self.bbox)} "
            f"rotate={self.rotate!r}>"
        )
