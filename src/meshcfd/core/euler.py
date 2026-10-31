"""Vectorized ideal-gas Euler states and numerical fluxes."""

from __future__ import annotations

import numpy as np


def primitive_to_conservative(
    primitive: np.ndarray, gamma: float
) -> np.ndarray:
    density, velocity_x, velocity_y, pressure = primitive.T
    momentum_x = density * velocity_x
    momentum_y = density * velocity_y
    energy = pressure / (gamma - 1.0) + 0.5 * density * (
        velocity_x**2 + velocity_y**2
    )
    return np.column_stack((density, momentum_x, momentum_y, energy))


def conservative_to_primitive(
    conservative: np.ndarray, gamma: float
) -> np.ndarray:
    density = conservative[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        velocity_x = conservative[:, 1] / density
        velocity_y = conservative[:, 2] / density
        pressure = (gamma - 1.0) * (
            conservative[:, 3]
            - 0.5
            * (conservative[:, 1] ** 2 + conservative[:, 2] ** 2)
            / density
        )
    primitive = np.column_stack((density, velocity_x, velocity_y, pressure))
    if (
        not np.isfinite(primitive).all()
        or np.any(density <= 0.0)
        or np.any(pressure <= 0.0)
    ):
        raise ValueError("Euler state has non-finite or non-positive density or pressure.")
    return primitive


def physical_flux(
    primitive: np.ndarray, normals: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    conservative = primitive_to_conservative(primitive, gamma)
    density, velocity_x, velocity_y, pressure = primitive.T
    normal_velocity = velocity_x * normals[:, 0] + velocity_y * normals[:, 1]
    flux = np.column_stack(
        (
            density * normal_velocity,
            density * velocity_x * normal_velocity + pressure * normals[:, 0],
            density * velocity_y * normal_velocity + pressure * normals[:, 1],
            (conservative[:, 3] + pressure) * normal_velocity,
        )
    )
    return flux, conservative


def numerical_flux(
    left: np.ndarray,
    right: np.ndarray,
    normals: np.ndarray,
    gamma: float,
    scheme: str,
) -> np.ndarray:
    if scheme == "Rusanov":
        return _rusanov_flux(left, right, normals, gamma)
    if scheme == "HLLC":
        return _hllc_flux(left, right, normals, gamma)
    if scheme == "Roe":
        return _roe_flux(left, right, normals, gamma)
    raise ValueError(f"Unsupported convective flux '{scheme}'.")


def _sound_speed(primitive: np.ndarray, gamma: float) -> np.ndarray:
    return np.sqrt(gamma * primitive[:, 3] / primitive[:, 0])


def _rusanov_flux(
    left: np.ndarray, right: np.ndarray, normals: np.ndarray, gamma: float
) -> np.ndarray:
    flux_left, state_left = physical_flux(left, normals, gamma)
    flux_right, state_right = physical_flux(right, normals, gamma)
    normal_left = np.sum(left[:, 1:3] * normals, axis=1)
    normal_right = np.sum(right[:, 1:3] * normals, axis=1)
    speed = np.maximum(
        np.abs(normal_left) + _sound_speed(left, gamma),
        np.abs(normal_right) + _sound_speed(right, gamma),
    )
    return 0.5 * (
        flux_left + flux_right - speed[:, None] * (state_right - state_left)
    )


def _hllc_flux(
    left: np.ndarray, right: np.ndarray, normals: np.ndarray, gamma: float
) -> np.ndarray:
    flux_left, state_left = physical_flux(left, normals, gamma)
    flux_right, state_right = physical_flux(right, normals, gamma)
    rho_left, rho_right = left[:, 0], right[:, 0]
    pressure_left, pressure_right = left[:, 3], right[:, 3]
    un_left = np.sum(left[:, 1:3] * normals, axis=1)
    un_right = np.sum(right[:, 1:3] * normals, axis=1)
    sound_left, sound_right = _sound_speed(left, gamma), _sound_speed(right, gamma)
    wave_left = np.minimum(un_left - sound_left, un_right - sound_right)
    wave_right = np.maximum(un_left + sound_left, un_right + sound_right)
    denominator = (
        rho_left * (wave_left - un_left)
        - rho_right * (wave_right - un_right)
    )
    safe_denominator = np.where(
        np.abs(denominator) > 1e-14, denominator, np.copysign(1e-14, denominator)
    )
    contact_speed = (
        pressure_right
        - pressure_left
        + rho_left * un_left * (wave_left - un_left)
        - rho_right * un_right * (wave_right - un_right)
    ) / safe_denominator

    def star_state(
        primitive: np.ndarray,
        conservative: np.ndarray,
        normal_velocity: np.ndarray,
        wave_speed: np.ndarray,
    ) -> np.ndarray:
        pressure = primitive[:, 3]
        factor = (
            primitive[:, 0]
            * (wave_speed - normal_velocity)
            / (wave_speed - contact_speed)
        )
        velocity = primitive[:, 1:3] + (
            contact_speed - normal_velocity
        )[:, None] * normals
        energy = factor * (
            conservative[:, 3] / primitive[:, 0]
            + (contact_speed - normal_velocity)
            * (
                contact_speed
                + pressure
                / (primitive[:, 0] * (wave_speed - normal_velocity))
            )
        )
        return np.column_stack(
            (factor, factor[:, None] * velocity, energy)
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        star_left = star_state(left, state_left, un_left, wave_left)
        star_right = star_state(right, state_right, un_right, wave_right)
        flux_star_left = flux_left + wave_left[:, None] * (star_left - state_left)
        flux_star_right = flux_right + wave_right[:, None] * (star_right - state_right)
    result = np.where(
        (wave_left >= 0.0)[:, None],
        flux_left,
        np.where(
            (contact_speed >= 0.0)[:, None],
            flux_star_left,
            np.where((wave_right > 0.0)[:, None], flux_star_right, flux_right),
        ),
    )
    if not np.isfinite(result).all():
        raise ValueError("HLLC flux produced a non-finite value.")
    return result


def _roe_flux(
    left: np.ndarray, right: np.ndarray, normals: np.ndarray, gamma: float
) -> np.ndarray:
    flux_left, _ = physical_flux(left, normals, gamma)
    flux_right, _ = physical_flux(right, normals, gamma)
    root_left = np.sqrt(left[:, 0])
    root_right = np.sqrt(right[:, 0])
    denominator = root_left + root_right
    velocity = (
        root_left[:, None] * left[:, 1:3]
        + root_right[:, None] * right[:, 1:3]
    ) / denominator[:, None]
    total_enthalpy = (
        root_left * (left[:, 3] / left[:, 0] + 0.5 * np.sum(left[:, 1:3] ** 2, axis=1))
        + root_right
        * (right[:, 3] / right[:, 0] + 0.5 * np.sum(right[:, 1:3] ** 2, axis=1))
    ) / denominator
    normal_velocity = np.sum(velocity * normals, axis=1)
    tangent_velocity = velocity[:, 0] * -normals[:, 1] + velocity[:, 1] * normals[:, 0]
    sound_squared = (gamma - 1.0) * (
        total_enthalpy - 0.5 * np.sum(velocity**2, axis=1)
    )
    if np.any(sound_squared <= 0.0) or not np.isfinite(sound_squared).all():
        raise ValueError("Roe average has a non-positive sound speed.")
    sound = np.sqrt(sound_squared)
    density_average = root_left * root_right

    un_left = np.sum(left[:, 1:3] * normals, axis=1)
    un_right = np.sum(right[:, 1:3] * normals, axis=1)
    ut_left = left[:, 1] * -normals[:, 1] + left[:, 2] * normals[:, 0]
    ut_right = right[:, 1] * -normals[:, 1] + right[:, 2] * normals[:, 0]
    delta_density = right[:, 0] - left[:, 0]
    delta_normal_velocity = un_right - un_left
    delta_tangent_velocity = ut_right - ut_left
    delta_pressure = right[:, 3] - left[:, 3]

    alpha_minus = (
        delta_pressure - density_average * sound * delta_normal_velocity
    ) / (2.0 * sound_squared)
    alpha_plus = (
        delta_pressure + density_average * sound * delta_normal_velocity
    ) / (2.0 * sound_squared)
    alpha_contact = delta_density - delta_pressure / sound_squared
    alpha_shear = density_average * delta_tangent_velocity
    lambdas = np.column_stack(
        (normal_velocity - sound, normal_velocity, normal_velocity, normal_velocity + sound)
    )
    entropy_fix = 0.1 * sound[:, None]
    absolute_lambdas = np.abs(lambdas)
    absolute_lambdas = np.where(
        absolute_lambdas < entropy_fix,
        0.5 * (absolute_lambdas**2 / entropy_fix + entropy_fix),
        absolute_lambdas,
    )
    eigenvectors = np.zeros((len(left), 4, 4))
    eigenvectors[:, :, 0] = np.column_stack(
        (np.ones(len(left)), normal_velocity - sound, tangent_velocity,
         total_enthalpy - normal_velocity * sound)
    )
    eigenvectors[:, :, 1] = np.column_stack(
        (
            np.ones(len(left)),
            normal_velocity,
            tangent_velocity,
            0.5 * (normal_velocity**2 + tangent_velocity**2),
        )
    )
    eigenvectors[:, :, 2] = np.column_stack(
        (np.zeros(len(left)), np.zeros(len(left)), np.ones(len(left)), tangent_velocity)
    )
    eigenvectors[:, :, 3] = np.column_stack(
        (np.ones(len(left)), normal_velocity + sound, tangent_velocity,
         total_enthalpy + normal_velocity * sound)
    )
    strengths = np.column_stack(
        (alpha_minus, alpha_contact, alpha_shear, alpha_plus)
    )
    dissipation_local = np.einsum(
        "nij,nj->ni", eigenvectors, absolute_lambdas * strengths
    )
    dissipation = np.column_stack(
        (
            dissipation_local[:, 0],
            dissipation_local[:, 1] * normals[:, 0]
            - dissipation_local[:, 2] * normals[:, 1],
            dissipation_local[:, 1] * normals[:, 1]
            + dissipation_local[:, 2] * normals[:, 0],
            dissipation_local[:, 3],
        )
    )
    result = 0.5 * (flux_left + flux_right - dissipation)
    if not np.isfinite(result).all():
        raise ValueError("Roe flux produced a non-finite value.")
    return result
