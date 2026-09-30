/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <data/TraceData.hpp>
#include <kernels/forward/rayTransition.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace hase::kernels::forward::ray
{
    using TriangleBarycentric = std::array<double, 3u>;

    /**
     * @param mesh Device trace containing triangular face connectivity.
     * @param cell Cell owning the face.
     * @param localFace Local face index.
     * @param point Cartesian point on the face plane.
     * @return Three affine face coordinates, or equal weights for a degenerate face.
     */
    [[nodiscard]] ALPAKA_FN_ACC inline TriangleBarycentric triangleBarycentricCoordinates(
        hase::data::TraceView const& mesh,
        unsigned const cell,
        unsigned const localFace,
        hase::core::Point const point)
    {
        auto const a = mesh.getPoint(static_cast<unsigned>(mesh.getCellFacePoint(cell, localFace, 0u)));
        auto const b = mesh.getPoint(static_cast<unsigned>(mesh.getCellFacePoint(cell, localFace, 1u)));
        auto const c = mesh.getPoint(static_cast<unsigned>(mesh.getCellFacePoint(cell, localFace, 2u)));
        auto const v0 = b - a;
        auto const v1 = c - a;
        auto const v2 = point - a;
        double const d00 = hase::core::dot(v0, v0);
        double const d01 = hase::core::dot(v0, v1);
        double const d11 = hase::core::dot(v1, v1);
        double const d20 = hase::core::dot(v2, v0);
        double const d21 = hase::core::dot(v2, v1);
        double const denominator = d00 * d11 - d01 * d01;
        if(alpaka::math::abs(denominator) <= std::numeric_limits<double>::epsilon())
            return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        double const second = (d11 * d20 - d01 * d21) / denominator;
        double const third = (d00 * d21 - d01 * d20) / denominator;
        return {1.0 - second - third, second, third};
    }

    /**
     * @param mesh Device trace containing triangular face connectivity.
     * @param cell Cell owning the face.
     * @param localFace Local face index.
     * @param coordinates Three face coordinates in local-vertex order.
     * @return Reconstructed Cartesian face position.
     */
    [[nodiscard]] ALPAKA_FN_ACC inline hase::core::Point positionFromTriangleBarycentric(
        hase::data::TraceView const& mesh,
        unsigned const cell,
        unsigned const localFace,
        TriangleBarycentric const coordinates)
    {
        hase::core::Point result{0.0, 0.0, 0.0};
        for(unsigned vertex = 0u; vertex < 3u; ++vertex)
            result = result
                     + mesh.getPoint(static_cast<unsigned>(mesh.getCellFacePoint(cell, localFace, vertex)))
                           * coordinates[vertex];
        return result;
    }

    namespace srmPosition
    {
        struct Centroid
        {
            static constexpr bool storesBarycentric = false;

            [[nodiscard]] ALPAKA_FN_ACC static hase::core::Point restore(
                hase::data::TraceView const& mesh,
                unsigned const cell,
                unsigned const localFace)
            {
                return faceCentroid(mesh, cell, localFace);
            }
        };

        struct Barycentric
        {
            static constexpr bool storesBarycentric = true;
        };

        inline constexpr Centroid centroid;
        inline constexpr Barycentric barycentric;
    } // namespace srmPosition

    template<typename T>
    concept SrmPositionPolicy = std::same_as<std::remove_cvref_t<T>, srmPosition::Centroid>
                                || std::same_as<std::remove_cvref_t<T>, srmPosition::Barycentric>;

    namespace behaviourDimension
    {
        struct Cell
        {
        };

        struct Boundary
        {
        };

        struct Failure
        {
        };
    } // namespace behaviourDimension

    namespace trait
    {
        template<typename T>
        struct IsCellBehaviour : std::bool_constant<std::derived_from<T, behaviourDimension::Cell>>
        {
        };

        template<typename T>
        struct IsBoundaryBehaviour : std::bool_constant<std::derived_from<T, behaviourDimension::Boundary>>
        {
        };

        template<typename T>
        struct IsFailureBehaviour : std::bool_constant<std::derived_from<T, behaviourDimension::Failure>>
        {
        };

    } // namespace trait

    template<typename T>
    inline constexpr bool isCellBehaviour_v = trait::IsCellBehaviour<std::remove_cvref_t<T>>::value;

    template<typename T>
    inline constexpr bool isBoundaryBehaviour_v = trait::IsBoundaryBehaviour<std::remove_cvref_t<T>>::value;

    template<typename T>
    inline constexpr bool isFailureBehaviour_v = trait::IsFailureBehaviour<std::remove_cvref_t<T>>::value;

    namespace concepts
    {
        template<typename T>
        concept CellBehaviour = isCellBehaviour_v<T>;

        template<typename T>
        concept BoundaryBehaviour = isBoundaryBehaviour_v<T>;

        template<typename T>
        concept FailureBehaviour = isFailureBehaviour_v<T>;

        template<typename T>
        concept BehaviourTerm = CellBehaviour<T> || BoundaryBehaviour<T> || FailureBehaviour<T>;

    } // namespace concepts

    /** SRM boundary tag. Its position policy determines the ray's optional storage. */
    template<SrmPositionPolicy T_PositionPolicy = srmPosition::Centroid>
    struct BoundaryPolicySrm : behaviourDimension::Boundary
    {
        using PositionPolicy = T_PositionPolicy;
        static inline constexpr PositionPolicy positionPolicy{};
    };

    template<typename T>
    concept SrmBoundaryBehaviour
        = concepts::BoundaryBehaviour<T> && requires { typename std::remove_cvref_t<T>::PositionPolicy; };

    /** @brief Whether a boundary policy stops or resumes the current walk. */
    enum class BoundaryResult : std::uint8_t
    {
        stop,
        continueTraversal
    };

    /** @brief Mutable ray state required by the generic Tet4 walker. */
    template<typename T_Ray>
    concept State = requires(T_Ray rayState) {
        rayState.position;
        rayState.direction;
        rayState.cell;
        rayState.forbiddenFace;
    };

    template<typename T_Term, typename T_Acc, typename T_RayState>
    concept HasOnCell = concepts::CellBehaviour<T_Term> && alpaka::onAcc::concepts::Acc<T_Acc> && State<T_RayState>
                        && requires(
                            T_Term& term,
                            T_Acc const& acc,
                            hase::data::TraceView const& mesh,
                            T_RayState& rayState,
                            unsigned const cell,
                            Tet4FaceIntersection const intersection) {
                               { term(acc, mesh, rayState, cell, intersection) } -> std::same_as<bool>;
                           };

    template<typename T_Term, typename T_Acc, typename T_RayState>
    concept HasOnBoundary
        = concepts::BoundaryBehaviour<T_Term> && alpaka::onAcc::concepts::Acc<T_Acc> && State<T_RayState>
          && requires(
              T_Term& term,
              T_Acc const& acc,
              hase::data::TraceView const& mesh,
              T_RayState& rayState,
              unsigned const cell,
              unsigned const localFace) {
                 { term(acc, mesh, rayState, cell, localFace) } -> std::same_as<BoundaryResult>;
             };

    template<typename T_Term, typename T_Acc, typename T_RayState>
    concept HasOnFailure
        = concepts::FailureBehaviour<T_Term> && alpaka::onAcc::concepts::Acc<T_Acc> && State<T_RayState>
          && requires(T_Term& term, T_Acc const& acc, hase::data::TraceView const& mesh, T_RayState& rayState) {
                 { term(acc, mesh, rayState) } -> std::same_as<void>;
             };

    inline constexpr BoundaryPolicySrm aseSrmPolicy{};
    inline constexpr BoundaryPolicySrm<srmPosition::Barycentric> pumpSrmPolicy{};

    struct NoSrmPositionStorage
    {
    };

    struct BarycentricSrmPositionStorage
    {
        TriangleBarycentric boundaryBarycentric{};
    };

    template<SrmPositionPolicy T_PositionPolicy>
    struct SrmPositionStorage : NoSrmPositionStorage
    {
    };

    template<>
    struct SrmPositionStorage<srmPosition::Barycentric> : BarycentricSrmPositionStorage
    {
    };

    ALPAKA_FN_ACC inline void captureSrmPosition(
        srmPosition::Centroid,
        hase::data::TraceView const&,
        unsigned,
        unsigned,
        hase::core::Point const,
        NoSrmPositionStorage&)
    {
    }

    ALPAKA_FN_ACC inline void captureSrmPosition(
        srmPosition::Barycentric,
        hase::data::TraceView const& mesh,
        unsigned const cell,
        unsigned const localFace,
        hase::core::Point const position,
        BarycentricSrmPositionStorage& storage)
    {
        storage.boundaryBarycentric = triangleBarycentricCoordinates(mesh, cell, localFace, position);
    }

    [[nodiscard]] ALPAKA_FN_ACC inline hase::core::Point restoreSrmPosition(
        srmPosition::Centroid,
        hase::data::TraceView const& mesh,
        unsigned const cell,
        unsigned const localFace)
    {
        return faceCentroid(mesh, cell, localFace);
    }

    [[nodiscard]] ALPAKA_FN_ACC inline hase::core::Point restoreSrmPosition(
        srmPosition::Barycentric,
        hase::data::TraceView const& mesh,
        unsigned const cell,
        unsigned const localFace,
        BarycentricSrmPositionStorage const& storage)
    {
        return positionFromTriangleBarycentric(mesh, cell, localFace, storage.boundaryBarycentric);
    }

    /** @brief Minimal geometry state for a ray traversing a Tet4 mesh. */
    struct TraversalState
    {
        hase::core::Point position{};
        hase::core::Point direction{};
        unsigned cell = 0u;
        std::int32_t forbiddenFace = -1;
    };

    /**
     * @brief Terminate a domain-local walk at an exterior boundary.
     *
     * Boundary behavior is deliberately separate from cell propagation. A
     * later scheduler can replace this policy with one that records a compact
     * ray handoff for another component without making the local walk recurse
     * into that component.
     */
    struct BoundaryPolicyEscape : behaviourDimension::Boundary
    {
        ALPAKA_FN_ACC BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const&,
            State auto&,
            unsigned,
            unsigned)
        {
            return BoundaryResult::stop;
        }
    };

    /**
     * Compile-time composition of ray-walk behaviour terms.
     *
     * Terms are callable objects categorized as cell or boundary behaviour.
     * Different categories can be supplied in any order; multiple terms in
     * one category are composed in the supplied order.
     */
    template<concepts::BehaviourTerm... T_Terms>
    struct RayWalkBehaviour : T_Terms...
    {
        static_assert((concepts::CellBehaviour<T_Terms> || ...), "ray walk requires cell behaviour");
        static_assert(
            (std::size_t{0u} + ... + static_cast<std::size_t>(concepts::BoundaryBehaviour<T_Terms>)) == 1u,
            "ray walk requires exactly one boundary behaviour");
        static_assert(
            (std::size_t{0u} + ... + static_cast<std::size_t>(concepts::FailureBehaviour<T_Terms>)) <= 1u,
            "ray walk accepts at most one failure behaviour");

        ALPAKA_FN_HOST_ACC constexpr RayWalkBehaviour(T_Terms... terms) : T_Terms{terms}...
        {
        }

        ALPAKA_FN_ACC bool onCell(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const& mesh,
            State auto& rayState,
            unsigned const cell,
            Tet4FaceIntersection const intersection)
        {
            return (invokeCell(static_cast<T_Terms&>(*this), acc, mesh, rayState, cell, intersection) && ...);
        }

        ALPAKA_FN_ACC BoundaryResult onBoundary(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const& mesh,
            State auto& rayState,
            unsigned const cell,
            unsigned const localFace)
        {
            bool const continueTraversal
                = ((invokeBoundary(static_cast<T_Terms&>(*this), acc, mesh, rayState, cell, localFace)
                    == BoundaryResult::continueTraversal)
                   && ...);
            return continueTraversal ? BoundaryResult::continueTraversal : BoundaryResult::stop;
        }

        ALPAKA_FN_ACC void onFailure(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const& mesh,
            State auto& rayState)
        {
            (invokeFailure(static_cast<T_Terms&>(*this), acc, mesh, rayState), ...);
        }

    private:
        template<concepts::CellBehaviour T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires HasOnCell<T_Term, T_Acc, T_RayState>
        ALPAKA_FN_ACC static bool invokeCell(
            T_Term& term,
            T_Acc const& acc,
            hase::data::TraceView const& mesh,
            T_RayState& rayState,
            unsigned const cell,
            Tet4FaceIntersection const intersection)
        {
            return term(acc, mesh, rayState, cell, intersection);
        }

        template<concepts::BehaviourTerm T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires(!HasOnCell<T_Term, T_Acc, T_RayState>)
        ALPAKA_FN_ACC static bool invokeCell(
            T_Term&,
            T_Acc const&,
            hase::data::TraceView const&,
            T_RayState&,
            unsigned,
            Tet4FaceIntersection)
        {
            return true;
        }

        template<concepts::BoundaryBehaviour T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires HasOnBoundary<T_Term, T_Acc, T_RayState>
        ALPAKA_FN_ACC static BoundaryResult invokeBoundary(
            T_Term& term,
            T_Acc const& acc,
            hase::data::TraceView const& mesh,
            T_RayState& rayState,
            unsigned const cell,
            unsigned const localFace)
        {
            return term(acc, mesh, rayState, cell, localFace);
        }

        template<concepts::BehaviourTerm T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires(!HasOnBoundary<T_Term, T_Acc, T_RayState>)
        ALPAKA_FN_ACC static BoundaryResult invokeBoundary(
            T_Term&,
            T_Acc const&,
            hase::data::TraceView const&,
            T_RayState&,
            unsigned,
            unsigned)
        {
            return BoundaryResult::continueTraversal;
        }

        template<concepts::FailureBehaviour T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires HasOnFailure<T_Term, T_Acc, T_RayState>
        ALPAKA_FN_ACC static void invokeFailure(
            T_Term& term,
            T_Acc const& acc,
            hase::data::TraceView const& mesh,
            T_RayState& rayState)
        {
            term(acc, mesh, rayState);
        }

        template<concepts::BehaviourTerm T_Term, alpaka::onAcc::concepts::Acc T_Acc, State T_RayState>
        requires(!HasOnFailure<T_Term, T_Acc, T_RayState>)
        ALPAKA_FN_ACC static void invokeFailure(T_Term&, T_Acc const&, hase::data::TraceView const&, T_RayState&)
        {
        }
    };

    template<typename... T_Terms>
    RayWalkBehaviour(T_Terms...) -> RayWalkBehaviour<T_Terms...>;

    /**
     * Walk one ray through the Tet4 mesh.
     *
     * Cell contribution and boundary handling are compile-time policies. The
     * only branches left here are geometric state transitions that every ray
     * tracer must perform.
     *
     * @param acc Accelerator context supplied to behavior terms.
     * @param mesh Device-resident Tet4 trace.
     * @param rayState Mutable position, direction, cell, and entry-face state.
     * @param behaviour Compile-time cell, boundary, and optional failure-policy composition.
     *
     * Traversal intentionally has no fixed step count: a valid physical path
     * can cross arbitrarily many cells as the mesh is refined. Failure policies
     * report geometric non-progress or invalid transitions without imposing a
     * mesh-resolution-dependent termination condition.
     */
    template<State T_Ray, alpaka::concepts::SpecializationOf<RayWalkBehaviour> T_Behaviour>
    ALPAKA_FN_ACC void walk(
        alpaka::onAcc::concepts::Acc auto const& acc,
        hase::data::TraceView const& mesh,
        T_Ray& rayState,
        T_Behaviour behaviour)
    {
        // A ray must make geometric progress: each accepted cell segment advances the
        // position by a positive length. A transition that leaves the position unchanged
        // therefore indicates non-progressing connectivity - for example a ray resting on
        // an exterior face whose boundary policy keeps requesting continueTraversal. After a
        // bounded number of consecutive position-preserving iterations, the walk reports a
        // traversal failure instead of spinning forever. This counts consecutive
        // non-progressing iterations, not total crossings, so it does not terminate a valid
        // ray as the mesh is refined.
        constexpr unsigned maxNonProgressingIterations = 1024u;
        unsigned nonProgressingIterations = 0u;
        hase::core::Point previousPosition = rayState.position;
        while(true)
        {
            bool const positionChanged = rayState.position.x != previousPosition.x
                                         || rayState.position.y != previousPosition.y
                                         || rayState.position.z != previousPosition.z;
            if(positionChanged)
            {
                nonProgressingIterations = 0u;
            }
            else if(++nonProgressingIterations > maxNonProgressingIterations)
            {
                behaviour.onFailure(acc, mesh, rayState);
                return;
            }
            previousPosition = rayState.position;

            unsigned const currentCell = rayState.cell;
            assert(currentCell < mesh.numberOfCells);
            auto const intersection = nextFaceIntersection(
                mesh,
                currentCell,
                rayState.position,
                rayState.direction,
                rayState.forbiddenFace);
            if(intersection.localFace < 0)
            {
                int const recoveryFace = isNearTet4Face(mesh, currentCell, rayState.position)
                                             ? immediateExitFace(
                                                   mesh,
                                                   currentCell,
                                                   rayState.position,
                                                   rayState.direction,
                                                   rayState.forbiddenFace)
                                             : -1;
                if(recoveryFace < 0)
                {
                    behaviour.onFailure(acc, mesh, rayState);
                    return;
                }
                auto const transition
                    = recoverFaceTransition(mesh, currentCell, recoveryFace, rayState.position, rayState.direction);
                if(transition.status == Tet4TransitionStatus::failed)
                {
                    rayState.cell = transition.cell;
                    behaviour.onFailure(acc, mesh, rayState);
                    return;
                }
                if(transition.status == Tet4TransitionStatus::reachedBoundary)
                {
                    rayState.cell = transition.cell;
                    auto const boundaryResult = behaviour.onBoundary(
                        acc,
                        mesh,
                        rayState,
                        transition.cell,
                        static_cast<unsigned>(transition.boundaryFace));
                    if(boundaryResult == BoundaryResult::continueTraversal)
                        continue;
                    return;
                }
                rayState.cell = transition.cell;
                rayState.forbiddenFace = transition.forbiddenFace;
                continue;
            }

            if(!behaviour.onCell(acc, mesh, rayState, currentCell, intersection))
            {
                return;
            }
            rayState.position = advance(rayState.position, rayState.direction, intersection.length);
            auto const transition
                = transitionAcrossIntersection(mesh, currentCell, intersection, rayState.position, rayState.direction);
            if(transition.status == Tet4TransitionStatus::failed)
            {
                rayState.cell = transition.cell;
                behaviour.onFailure(acc, mesh, rayState);
                return;
            }
            if(transition.status == Tet4TransitionStatus::reachedBoundary)
            {
                rayState.cell = transition.cell;
                auto const boundaryResult = behaviour.onBoundary(
                    acc,
                    mesh,
                    rayState,
                    transition.cell,
                    static_cast<unsigned>(transition.boundaryFace));
                if(boundaryResult == BoundaryResult::continueTraversal)
                    continue;
                return;
            }
            rayState.cell = transition.cell;
            rayState.forbiddenFace = transition.forbiddenFace;
        }
    }
} // namespace hase::kernels::forward::ray
