# from datetime import date

from pydantic import BaseModel, ConfigDict


class PostsBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str
    content: str


class PostsCreate(PostsBase):
    user_id: int


# FIXME: 'Datetimes provided to dates should have zero time - e.g. be exact dates', 'input': datetime.datetime(2024, 1, 11, 22, 16, 57), 'url': 'https://errors.pydantic.dev/2.5/v/date_from_datetime_inexact'}


class Posts(PostsBase):
    id: int
    # creation_date: date
    user_id: int
