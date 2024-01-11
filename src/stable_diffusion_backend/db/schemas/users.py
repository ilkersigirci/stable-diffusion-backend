# from datetime import date

from pydantic import BaseModel, ConfigDict, EmailStr, Field, constr

from stable_diffusion_backend.db.schemas.posts import Posts


class UsersBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    first_name: constr(to_lower=True)  # type: ignore
    last_name: constr(to_lower=True)  # type: ignore
    email: EmailStr
    is_admin: bool


class UsersCreate(UsersBase):
    password: str = Field(alias="password")


# FIXME: 'Datetimes provided to dates should have zero time - e.g. be exact dates', 'input': datetime.datetime(2024, 1, 11, 22, 16, 57), 'url': 'https://errors.pydantic.dev/2.5/v/date_from_datetime_inexact'}
class Users(UsersBase):
    id: int
    # creation_date: date
    posts: list[Posts] = []
