from typing import Annotated, Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from stable_diffusion_backend.db import sessions
from stable_diffusion_backend.db.models import Posts, Users
from stable_diffusion_backend.db.schemas import posts as posts_schema
from stable_diffusion_backend.deps import get_current_user

router = APIRouter(prefix="/posts", tags=["posts"])

CurrentUser = Annotated[Users, Depends(get_current_user)]
CurrentAsyncSession = Annotated[AsyncSession, Depends(sessions.get_async_session)]


@router.get("/get/posts")
async def get_posts(
    current_user: CurrentUser,
    db: CurrentAsyncSession,
) -> Sequence[posts_schema.Posts]:
    q = select(Posts)
    result = await db.execute(q)
    posts = result.scalars().all()
    if not posts:
        raise HTTPException(status_code=404, detail="No posts found")
    return posts


@router.get("/get/post/{id}")
async def get_post(
    current_user: CurrentUser,
    id: int,
    db: CurrentAsyncSession,
) -> posts_schema.Posts | dict:
    # Construct the query, in this case a scaler since we expect a single value
    # and don't need a tuple
    q = await db.scalars(select(Posts).filter(Posts.id == id))
    post = q.first()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.post("/add/post")
async def add_user(
    current_user: CurrentUser,
    post: posts_schema.PostsCreate,
    db: CurrentAsyncSession,
) -> str:
    posts_db = Posts(**post.dict())
    db.add(posts_db)
    await db.commit()
    return "ok"


# TODO: Implement post deleting functionality only for the user who created the post
@router.delete("/delete/post/{id}")
async def delete_user(
    current_user: CurrentUser,
    id: int,
    db: CurrentAsyncSession,
) -> str:
    q = await db.scalars(select(Posts).filter(Posts.id == id))
    post = q.first()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if post.id != current_user.id:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to delete other user's their posts",
        )

    q = delete(Posts).filter(Posts.id == id)
    await db.execute(q)
    await db.commit()
    return "ok"
