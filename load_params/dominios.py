from enum import Enum

# =========================
# 1) Definiciones de dominio
# =========================
class DayType(Enum):
    NORMAL = "normal"
    OFERTA = "oferta"
    DOMINGO = "domingo"


class LaneType(Enum):
    REGULAR = "regular"
    EXPRESS = "express"          # máx 10 ítems
    PRIORITY = "priority"
    SELF = "self_checkout"       # máx 15 ítems


class Profile(Enum):
    WEEKLY = "weekly_planner"
    REGULAR = "regular"
    FAMILY = "family_cart"
    DEAL = "deal_hunter"
    EXPRESS = "express_basket"
    SELF = "self_checkout_fan"


class Pay(Enum):
    CARD = "card"
    CASH = "cash"


class PriorityFlag(Enum):
    P = "pregnant"
    NP = "no_priority"
    S = "senior"
    RM = "reduced_mobility"
