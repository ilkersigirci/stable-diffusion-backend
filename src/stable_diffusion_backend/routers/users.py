import logging
from typing import Annotated, Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from stable_diffusion_backend.db import sessions
from stable_diffusion_backend.db.models import Users
from stable_diffusion_backend.db.schemas import users as user_schemas
from stable_diffusion_backend.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])

CurrentUser = Annotated[Users, Depends(get_current_user)]
CurrentAsyncSession = Annotated[AsyncSession, Depends(sessions.get_async_session)]


@router.get("/get/users")
async def get_users(
    current_user: CurrentUser,
    db: CurrentAsyncSession,
) -> Sequence[user_schemas.Users]:
    q = select(Users)
    result = await db.execute(q)
    users = result.scalars().all()

    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    return users


@router.get("/get/user/{id}")
async def get_user(
    current_user: CurrentUser,
    id: int,
    db: CurrentAsyncSession,
) -> user_schemas.Users | dict:
    # Construct the query, in this case a scaler since we expect a single value
    # and don't need a tuple
    q = await db.scalars(select(Users).filter(Users.id == id))
    user = q.first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# TODO: Add unit tests
@router.delete("/delete/user/{id}")
async def delete_user(
    current_user: CurrentUser,
    id: int,
    db: CurrentAsyncSession,
) -> str:
    q = await db.scalars(select(Users).filter(Users.id == id))
    user = q.first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_user.id:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to delete yourself",
        )
    logger.info(current_user.is_admin)

    if not current_user.is_admin:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are not an admin",
        )

    q = delete(Users).filter(Users.id == id)
    await db.execute(q)
    await db.commit()
    return "ok"
