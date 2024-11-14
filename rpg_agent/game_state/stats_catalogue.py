import weave
from pydantic import BaseModel
from rpg_agent.llm_predictor import LLMPredictor


class CharacterBaseStats(BaseModel):
    strength: int
    finesse: int
    intelligence: int
    constitution: int
    memory: int
    wits: int


class CharacterCombatStats(BaseModel):
    damage_max: int
    damage_min: int
    critical_chance: float
    accuracy: float
    dodging: float
    physical_armour_current: int
    physical_armour_total: int
    magic_armour_current: int
    magic_armour_total: int


class CharacterActionStats(BaseModel):
    movement: int
    initiative: int
    experience: int
    next_level: int


class CharacterElementalStats(BaseModel):
    fire: int
    water: int
    earth: int
    air: int
    poison: int


class CharacterHealthStats(BaseModel):
    vitality_total: int
    vitality_current: int
    action_points: int
    source_points: int


class CharacterAttributes(BaseModel):
    base_stats: CharacterBaseStats
    combat_stats: CharacterCombatStats
    action_stats: CharacterActionStats
    elemental_stats: CharacterElementalStats
    health_stats: CharacterHealthStats


class Coordinates(BaseModel):
    x: float
    y: float


class CharacterEquipmentCoordinates(BaseModel):
    coordinates: list[Coordinates]
