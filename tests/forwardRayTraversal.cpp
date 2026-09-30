#include <alpaka/math.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <alpakaUtils/memory.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <data/TraceData.hpp>
#include <kernels/forward/policyRay.hpp>
#include <kernels/forward/rayTransition.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <vector>

namespace hase::tests
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    hase::data::TraceData makeTraversalMesh(
        std::vector<hase::core::Point> const& points,
        std::vector<std::array<unsigned, hase::data::tet4VertexCount>> const& cells)
    {
        hase::data::TraceData mesh;
        mesh.numberOfCells = static_cast<unsigned>(cells.size());
        mesh.numberOfMeshPoints = static_cast<unsigned>(points.size());
        mesh.numberOfCellVertices = hase::data::tet4VertexCount;
        mesh.numberOfFacesPerCell = hase::data::tet4FaceCount;
        mesh.points.resize(points.size() * 3u);
        for(unsigned point = 0u; point < points.size(); ++point)
        {
            mesh.points[point] = points[point].x;
            mesh.points[point + points.size()] = points[point].y;
            mesh.points[point + 2u * points.size()] = points[point].z;
        }

        using Face = std::array<unsigned, hase::data::tet4FaceWidth>;
        std::vector<Face> faces;
        faces.reserve(cells.size() * hase::data::tet4FaceCount);
        for(auto const& cell : cells)
        {
            mesh.cellPointIndices.insert(mesh.cellPointIndices.end(), cell.cbegin(), cell.cend());
            for(unsigned localFace = 0u; localFace < hase::data::tet4FaceCount; ++localFace)
            {
                Face face{};
                unsigned faceVertex = 0u;
                for(unsigned localVertex = 0u; localVertex < hase::data::tet4VertexCount; ++localVertex)
                {
                    if(localVertex != localFace)
                    {
                        face[faceVertex++] = cell[localVertex];
                    }
                }
                std::ranges::sort(face);
                faces.push_back(face);
                mesh.cellFaces.insert(mesh.cellFaces.end(), face.cbegin(), face.cend());
            }
        }

        mesh.cellNeighborCells.assign(faces.size(), -1);
        mesh.cellNeighborLocalFaces.assign(faces.size(), -1);
        for(unsigned face = 0u; face < faces.size(); ++face)
        {
            for(unsigned candidate = face + 1u; candidate < faces.size(); ++candidate)
            {
                if(faces[face] != faces[candidate])
                {
                    continue;
                }
                unsigned const cell = face / hase::data::tet4FaceCount;
                unsigned const localFace = face % hase::data::tet4FaceCount;
                unsigned const neighbor = candidate / hase::data::tet4FaceCount;
                unsigned const neighborLocalFace = candidate % hase::data::tet4FaceCount;
                mesh.cellNeighborCells[face] = static_cast<int>(neighbor);
                mesh.cellNeighborLocalFaces[face] = static_cast<int>(neighborLocalFace);
                mesh.cellNeighborCells[candidate] = static_cast<int>(cell);
                mesh.cellNeighborLocalFaces[candidate] = static_cast<int>(localFace);
            }
        }
        mesh.precomputeBarycentricFacePlanes();
        return mesh;
    }

    hase::data::TraceView traversalView(hase::data::TraceData const& mesh)
    {
        hase::data::TraceView view{};
        view.points = mesh.points;
        view.cellPointIndices = mesh.cellPointIndices;
        view.cellFaces = mesh.cellFaces;
        view.barycentricFacePlanes = mesh.barycentricFacePlanes;
        view.cellNeighborCells = mesh.cellNeighborCells;
        view.cellNeighborLocalFaces = mesh.cellNeighborLocalFaces;
        view.numberOfCells = mesh.numberOfCells;
        view.numberOfFacesPerCell = mesh.numberOfFacesPerCell;
        view.numberOfCellVertices = mesh.numberOfCellVertices;
        view.numberOfMeshPoints = mesh.numberOfMeshPoints;
        return view;
    }

    struct TraversalResult
    {
        hase::kernels::forward::Tet4FaceIntersection intersection;
        hase::core::Point intersectionPoint;
        hase::kernels::forward::Tet4FaceTransition transition;
        hase::kernels::forward::Tet4FaceIntersection nextIntersection;
    };

    static_assert(std::is_trivially_copyable_v<TraversalResult>);

    struct TraverseOneRay
    {
        template<typename TAcc, alpaka::concepts::IMdSpan TResult>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            hase::data::TraceView const mesh,
            unsigned const startCell,
            hase::core::Point const origin,
            hase::core::Point const direction,
            TResult result) const
        {
            TraversalResult traversal;
            traversal.intersection
                = hase::kernels::forward::nextFaceIntersection(mesh, startCell, origin, direction, -1);
            traversal.intersectionPoint
                = hase::kernels::forward::advance(origin, direction, traversal.intersection.length);
            traversal.transition = hase::kernels::forward::transitionAcrossIntersection(
                mesh,
                startCell,
                traversal.intersection,
                traversal.intersectionPoint,
                direction);
            if(traversal.transition.status == hase::kernels::forward::Tet4TransitionStatus::enteredCell)
            {
                traversal.nextIntersection = hase::kernels::forward::nextFaceIntersection(
                    mesh,
                    traversal.transition.cell,
                    traversal.intersectionPoint,
                    direction,
                    traversal.transition.forbiddenFace);
            }
            result[0u] = traversal;
        }
    };

    TraversalResult traverseOneRay(
        auto& device,
        auto const executor,
        hase::data::TraceData& hostMesh,
        unsigned const startCell,
        hase::core::Point const origin,
        hase::core::Point const direction)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto deviceMesh = hostMesh.makeResident(device);
        deviceMesh.toDevice(queue);
        auto result = hase::alpakaUtils::getHybridBuffer(device, std::vector<TraversalResult>(1u));
        auto const frameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(device, executor, alpaka::Vec{1u});
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{
                TraverseOneRay{},
                deviceMesh.view(),
                startCell,
                origin,
                direction,
                result.toDeviceView()});
        result.toHost(queue);
        return result.getHostView()[0u];
    }

    /** Minimal single-cell walk that keeps resuming after an exterior boundary hit. */
    struct AlwaysContinueBoundary : hase::kernels::forward::ray::behaviourDimension::Boundary
    {
        ALPAKA_FN_ACC hase::kernels::forward::ray::BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const&,
            hase::kernels::forward::ray::State auto&,
            unsigned,
            unsigned) const
        {
            return hase::kernels::forward::ray::BoundaryResult::continueTraversal;
        }
    };

    /** Minimal cell policy that keeps traversing without depositing a contribution. */
    struct ContinueAtCell : hase::kernels::forward::ray::behaviourDimension::Cell
    {
        ALPAKA_FN_ACC bool operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const&,
            hase::kernels::forward::ray::State auto&,
            unsigned,
            hase::kernels::forward::Tet4FaceIntersection) const
        {
            return true;
        }
    };

    /** Reuses the generic walker's failure hook to record that the guard terminated the walk. */
    template<typename T_FailureView>
    struct RecordNonProgressFailure : hase::kernels::forward::ray::behaviourDimension::Failure
    {
        T_FailureView failures;

        ALPAKA_FN_HOST_ACC constexpr explicit RecordNonProgressFailure(T_FailureView value) : failures{value}
        {
        }

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const&,
            hase::kernels::forward::ray::State auto&)
        {
            alpaka::onAcc::atomicAdd(acc, &failures[0u], 1u);
        }
    };

    struct WalkOneRay
    {
        template<typename TAcc, alpaka::concepts::IMdSpan TIntFailure>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            hase::data::TraceView const mesh,
            unsigned const startCell,
            hase::core::Point const origin,
            hase::core::Point const direction,
            TIntFailure failure,
            bool const continueAtBoundary) const
        {
            namespace ray = hase::kernels::forward::ray;
            ray::TraversalState rayState;
            rayState.position = origin;
            rayState.direction = direction;
            rayState.cell = startCell;
            rayState.forbiddenFace = -1;
            if(continueAtBoundary)
            {
                ray::walk(
                    acc,
                    mesh,
                    rayState,
                    ray::RayWalkBehaviour{
                        ContinueAtCell{},
                        AlwaysContinueBoundary{},
                        RecordNonProgressFailure<TIntFailure>{failure}});
            }
            else
            {
                ray::walk(
                    acc,
                    mesh,
                    rayState,
                    ray::RayWalkBehaviour{
                        ContinueAtCell{},
                        ray::BoundaryPolicyEscape{},
                        RecordNonProgressFailure<TIntFailure>{failure}});
            }
        }
    };

    std::uint32_t walkRay(
        auto& device,
        auto const executor,
        hase::data::TraceData& hostMesh,
        unsigned const startCell,
        hase::core::Point const origin,
        hase::core::Point const direction,
        bool const continueAtBoundary)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto deviceMesh = hostMesh.makeResident(device);
        deviceMesh.toDevice(queue);
        auto failure = hase::alpakaUtils::getHybridBuffer(device, std::vector<std::uint32_t>(1u, 0u));
        failure.toDevice(queue);
        auto const frameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(device, executor, alpaka::Vec{1u});
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{
                WalkOneRay{},
                deviceMesh.view(),
                startCell,
                origin,
                direction,
                failure.toDeviceView(),
                continueAtBoundary});
        failure.toHost(queue);
        return failure.getHostView()[0u];
    }

} // namespace hase::tests

TEMPLATE_LIST_TEST_CASE(
    "forward ray crosses a vertex shared by 30 tetrahedra",
    "[forward][traversal][edge-case]",
    hase::tests::TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);

    constexpr unsigned tetrahedraPerHalf = 15u;
    constexpr unsigned sharedVertex = 0u;
    constexpr double pi = 3.14159265358979323846;
    std::vector<hase::core::Point> points{{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
    for(unsigned point = 0u; point < tetrahedraPerHalf; ++point)
    {
        double const angle = 2.0 * pi * static_cast<double>(point) / tetrahedraPerHalf;
        points.push_back({std::cos(angle), std::sin(angle), 0.0});
    }

    std::vector<std::array<unsigned, hase::data::tet4VertexCount>> cells;
    cells.reserve(2u * tetrahedraPerHalf);
    for(unsigned cell = 0u; cell < tetrahedraPerHalf; ++cell)
    {
        unsigned const ring0 = 3u + cell;
        unsigned const ring1 = 3u + (cell + 1u) % tetrahedraPerHalf;
        cells.push_back({sharedVertex, 1u, ring0, ring1});
    }
    for(unsigned cell = 0u; cell < tetrahedraPerHalf; ++cell)
    {
        unsigned const ring0 = 3u + cell;
        unsigned const ring1 = 3u + (cell + 1u) % tetrahedraPerHalf;
        cells.push_back({sharedVertex, 2u, ring1, ring0});
    }
    auto mesh = hase::tests::makeTraversalMesh(points, cells);

    CHECK(mesh.numberOfCells == 30u);
    CHECK(
        std::ranges::count(mesh.cellPointIndices, sharedVertex)
        == static_cast<std::ranges::range_difference_t<decltype(mesh.cellPointIndices)>>(mesh.numberOfCells));

    hase::core::Point const origin = points[1u] * 0.2 + points[3u] * 0.1 + points[4u] * 0.3;
    hase::core::Point const direction = hase::kernels::forward::normalize(origin * -1.0);
    auto const result = hase::tests::traverseOneRay(device, alpaka::getExecutor(backend), mesh, 0u, origin, direction);

    CHECK(hase::kernels::forward::hasMultipleTiedFaces(result.intersection.tiedFaceMask));
    CHECK(result.intersection.tiedFaceMask == ((1u << 1u) | (1u << 2u) | (1u << 3u)));
    CHECK(result.intersectionPoint.x == Catch::Approx(0.0).margin(1.0e-14));
    CHECK(result.intersectionPoint.y == Catch::Approx(0.0).margin(1.0e-14));
    CHECK(result.intersectionPoint.z == Catch::Approx(0.0).margin(1.0e-14));
    REQUIRE(result.transition.status == hase::kernels::forward::Tet4TransitionStatus::enteredCell);
    CHECK(result.transition.cell == 23u);
    CHECK(result.nextIntersection.localFace >= 0);
    CHECK(result.nextIntersection.length > 0.0);
}

TEMPLATE_LIST_TEST_CASE(
    "forward ray crosses a plane at perfectly perpendicular incidence",
    "[forward][traversal][edge-case]",
    hase::tests::TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);

    std::vector<hase::core::Point> const
        points{{-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}};
    auto mesh = hase::tests::makeTraversalMesh(points, {{0u, 1u, 2u, 3u}, {4u, 1u, 3u, 2u}});

    hase::core::Point const direction{1.0, 0.0, 0.0};
    hase::core::Point const planeEdge0 = points[2u] - points[1u];
    hase::core::Point const planeEdge1 = points[3u] - points[1u];
    CHECK(hase::core::dot(direction, planeEdge0) == 0.0);
    CHECK(hase::core::dot(direction, planeEdge1) == 0.0);

    hase::core::Point const origin{-0.4, 0.2, 0.2};
    auto const result = hase::tests::traverseOneRay(device, alpaka::getExecutor(backend), mesh, 0u, origin, direction);

    CHECK(result.intersection.localFace == 0);
    CHECK(result.intersection.tiedFaceMask == 1u);
    CHECK(result.intersection.length == Catch::Approx(0.4));
    CHECK(result.intersectionPoint.x == Catch::Approx(0.0).margin(1.0e-14));
    CHECK(result.intersectionPoint.y == Catch::Approx(0.2));
    CHECK(result.intersectionPoint.z == Catch::Approx(0.2));
    REQUIRE(result.transition.status == hase::kernels::forward::Tet4TransitionStatus::enteredCell);
    CHECK(result.transition.cell == 1u);
    CHECK(result.transition.forbiddenFace == 0);
    CHECK(result.nextIntersection.localFace >= 0);
    CHECK(result.nextIntersection.length == Catch::Approx(0.6));
}

TEMPLATE_LIST_TEST_CASE(
    "forward ray on an exterior face reports failure instead of spinning",
    "[forward][traversal][edge-case]",
    hase::tests::TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);

    // Single tetrahedron. Face 0 spans vertices {1,2,3}, which lie in the plane x = 0,
    // and is the exterior tet4BoundaryFace. The opposite vertex 0 sits at x = 0.5, so the
    // outward direction for face 0 is -x.
    std::vector<hase::core::Point> const points{{0.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
    auto mesh = hase::tests::makeTraversalMesh(points, {{0u, 1u, 2u, 3u}});

    hase::core::Point const direction{-1.0, 0.0, 0.0};
    auto const hostView = hase::tests::traversalView(mesh);

    // A normal onward ray advances and terminates at the escaping exterior boundary without
    // tripping the non-progress guard.
    hase::core::Point const interiorOrigin{0.125, 0.25, 0.0};
    auto const escaped = hase::tests::walkRay(
        device,
        alpaka::getExecutor(backend),
        mesh,
        0u,
        interiorOrigin,
        direction,
        /*continueAtBoundary=*/false);
    CHECK(escaped == 0u);

    // A ray resting on face 0 moving outward produces no cell segment
    // (nextFaceIntersection finds no forward face), so the recovery path re-selects face 0
    // as an exterior boundary. A continueTraversal boundary policy would otherwise spin
    // forever; the guard must report a traversal failure instead.
    hase::core::Point const boundaryOrigin{0.0, 1.0 / 3.0, 0.0};
    REQUIRE(std::abs(hostView.getFaceBarycentricCoordinate(0u, 0u, boundaryOrigin)) <= 1.0e-12);
    REQUIRE(hostView.getFaceBarycentricDirection(0u, 0u, direction) < 0.0);
    auto const failed = hase::tests::walkRay(
        device,
        alpaka::getExecutor(backend),
        mesh,
        0u,
        boundaryOrigin,
        direction,
        /*continueAtBoundary=*/true);
    CHECK(failed == 1u);
}
