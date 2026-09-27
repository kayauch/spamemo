from pydantic import BaseModel, Field


class Detection2D(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
