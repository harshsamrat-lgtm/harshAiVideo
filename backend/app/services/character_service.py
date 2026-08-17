"""
Character Bible service for persistent identity, appearance, face references, voice profile binding, and LoRAs.
"""
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.database.models import CharacterModel
from app.models.schemas import CharacterCreate, CharacterResponse
from app.core.logging import logger


class CharacterService:
    @staticmethod
    async def create_character(project_id: Optional[str], data: CharacterCreate) -> CharacterResponse:
        char_id = f"char_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            char = CharacterModel(
                character_id=char_id,
                project_id=project_id,
                name=data.name,
                description=data.description,
                appearance=data.appearance,
                face_reference=data.face_reference,
                body_description=data.body_description,
                hair=data.hair,
                clothing=data.clothing,
                age_or_look=data.age_or_look,
                accessories=data.accessories,
                default_prompt=data.default_prompt,
                negative_prompt=data.negative_prompt,
                voice_profile_id=data.voice_profile_id,
                consistency_settings=data.consistency_settings or {},
                reference_images=data.reference_images or [],
                optional_lora=data.optional_lora,
                created_at=now,
                updated_at=now
            )
            session.add(char)
            await session.commit()
            await session.refresh(char)
            
            logger.info(f"Registered character in Bible: '{data.name}' (ID: {char_id}, Voice: {data.voice_profile_id})")
            return CharacterResponse(
                character_id=char.character_id,
                project_id=char.project_id,
                name=char.name,
                description=char.description,
                appearance=char.appearance,
                face_reference=char.face_reference,
                body_description=char.body_description,
                hair=char.hair,
                clothing=char.clothing,
                age_or_look=char.age_or_look,
                accessories=char.accessories,
                default_prompt=char.default_prompt,
                negative_prompt=char.negative_prompt,
                voice_profile_id=char.voice_profile_id,
                consistency_settings=char.consistency_settings or {},
                reference_images=char.reference_images or [],
                optional_lora=char.optional_lora,
                created_at=char.created_at,
                updated_at=char.updated_at
            )

    @staticmethod
    async def list_characters(project_id: Optional[str] = None) -> List[CharacterResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(CharacterModel)
            if project_id:
                stmt = stmt.where(
                    (CharacterModel.project_id == project_id) | (CharacterModel.project_id.is_(None))
                )
            result = await session.execute(stmt)
            chars = result.scalars().all()
            return [CharacterResponse(
                character_id=c.character_id,
                project_id=c.project_id,
                name=c.name,
                description=c.description,
                appearance=c.appearance,
                face_reference=c.face_reference,
                body_description=c.body_description,
                hair=c.hair,
                clothing=c.clothing,
                age_or_look=c.age_or_look,
                accessories=c.accessories,
                default_prompt=c.default_prompt,
                negative_prompt=c.negative_prompt,
                voice_profile_id=c.voice_profile_id,
                consistency_settings=c.consistency_settings or {},
                reference_images=c.reference_images or [],
                optional_lora=c.optional_lora,
                created_at=c.created_at,
                updated_at=c.updated_at
            ) for c in chars]

    @staticmethod
    async def get_character(character_id: str) -> Optional[CharacterResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(CharacterModel).where(CharacterModel.character_id == character_id)
            result = await session.execute(stmt)
            c = result.scalar_one_or_none()
            if not c:
                return None
            return CharacterResponse(
                character_id=c.character_id,
                project_id=c.project_id,
                name=c.name,
                description=c.description,
                appearance=c.appearance,
                face_reference=c.face_reference,
                body_description=c.body_description,
                hair=c.hair,
                clothing=c.clothing,
                age_or_look=c.age_or_look,
                accessories=c.accessories,
                default_prompt=c.default_prompt,
                negative_prompt=c.negative_prompt,
                voice_profile_id=c.voice_profile_id,
                consistency_settings=c.consistency_settings or {},
                reference_images=c.reference_images or [],
                optional_lora=c.optional_lora,
                created_at=c.created_at,
                updated_at=c.updated_at
            )


character_service = CharacterService()
