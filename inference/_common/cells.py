# Cell bookkeeping shared by wrappers whose models were trained on primitive cells.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

# Lattice centering -> lattice points per conventional cell. R assumes Z is quoted for the
# hexagonal axes (the usual convention); for rhombohedral axes pass Z = Z_rh * 3.
CENTERING = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}


def lattice_letter(spacegroup: str) -> str:
    """Centering letter of a Hermann-Mauguin symbol (P4mm) or space-group number (99)."""
    symbol = str(spacegroup).strip()
    if symbol.isdigit():
        from pymatgen.symmetry.groups import SpaceGroup

        if not 1 <= int(symbol) <= 230:
            raise SystemExit(f"Space-group number must be 1-230, got {symbol}")
        symbol = SpaceGroup.from_int_number(int(symbol)).symbol
    letter = symbol[:1].upper()
    if letter not in CENTERING:
        raise SystemExit(f"Unrecognized space group {spacegroup!r}; give a Hermann-Mauguin symbol (e.g. P4mm) or a number")
    return letter


def primitive_z_candidates(z: int, spacegroup: str | None) -> list[int]:
    """Formula units in the primitive cell for a conventional-cell Z.

    With a space group the conversion is exact; without one every centering that divides Z is
    returned, so callers can try each hypothesis separately.
    """
    if spacegroup:
        m = CENTERING[lattice_letter(spacegroup)]
        if z % m:
            raise SystemExit(f"Z={z} is not divisible by the {m} lattice points of {spacegroup}")
        return [z // m]
    return sorted({z // m for m in (1, 2, 3, 4) if z % m == 0})


def atom_list(composition: str, z: int) -> list[str]:
    """Element symbols, one per atom, for composition x z; rejects non-integer atom counts."""
    from pymatgen.core import Composition

    comp = Composition(composition) * z
    if any(abs(n - round(n)) > 1e-6 for n in comp.values()):
        raise SystemExit(f"{composition} x {z} is not an integer atom list; partial occupancies are not supported")
    return sorted(el.symbol for el, n in comp.items() for _ in range(int(round(n))))
