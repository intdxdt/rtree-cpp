#define CATCH_CONFIG_MAIN

#include <cmath>
#include <vector>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include "include/catch.h"
#include "include/mbr.h"
#include "rtree.h"


double get_time_nano() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
}

mbr::MBR<double> RandBox(double size, const std::function<double()> &rnd) {
    auto x = rnd() * (100.0 - size);
    auto y = rnd() * (100.0 - size);
    return {x, y, x + size * rnd(), y + size * rnd()};
}

std::vector<rtree::Item<double>> GenDataItems(size_t N, double size, size_t id = 0) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto rnd = [&]() {
        return distribution(generator);
    };

    std::vector<rtree::Item<double>> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        data.emplace_back(rtree::Item{id++, RandBox(size, rnd)});
    }
    return data;
}


namespace bench {
    auto N = size_t(1e6);
    auto maxFill = 64UL;
    std::vector<rtree::Item<double>> BenchData;//= GenDataItems(N, 1);
    std::vector<rtree::Item<double>> bboxes100;//= GenDataItems(1000, 100 * std::sqrt(0.1));
    std::vector<rtree::Item<double>> bboxes10;//= GenDataItems(1000, 10);
    std::vector<rtree::Item<double>> bboxes1;//= GenDataItems(1000, 1);
    double box_area{0};
    int foundTotal{0};

    void gen_bench_data() {
        std::cout << "Generating Bench Data\n";
        BenchData = GenDataItems(N, 1);
        bboxes100 = GenDataItems(1000, 100 * std::sqrt(0.1));
        bboxes10 = GenDataItems(1000, 10);
        bboxes1 = GenDataItems(1000, 1);
        std::cout << "Done ! Generating Bench Data\n";
    }

    void run_bench(std::string name, const std::function<void()> &func) {
        auto BN = 100UL;
        double times{0};
        for (size_t i = 0; i < 5; i++) { func(); }
        for (size_t i = 0; i < BN; i++) {
            auto t0 = get_time_nano();
            func();
            auto t1 = get_time_nano();
            times += (t1 - t0);
        }
        std::cout << "Bench : " << name << " : " << times / BN << " nano\n";
    }

    void Benchmark_Insert_OneByOne_SmallBigData() {
        auto tree = rtree::NewRTree<double>(maxFill);
        auto n = BenchData.size();
        for (size_t i = 0; i < n; i++) {
            tree.insert(BenchData[i]);
        }
        box_area = tree.data.bbox.bbox().area();
    }


    void Benchmark_Load_Data() {
        auto tree = rtree::NewRTree<double>(maxFill);
        tree.load(BenchData);
        box_area = tree.data.bbox.bbox().area();
    }

    TEST_CASE("bench-data", "[bench 1]") {
    }

    TEST_CASE("bench0", "[bench 0]") {
//        gen_bench_data();
//        run_bench("Benchmark_Insert_OneByOne_SmallBigData", Benchmark_Insert_OneByOne_SmallBigData);
    }

    TEST_CASE("bench1", "[bench 1]") {
//        gen_bench_data();
//        run_bench("Benchmark_Load_Data", Benchmark_Load_Data);
    }
}

namespace rtest {
    template<typename K, typename V>
    inline bool has_key(const std::map<K, V> &dict, const K &x) {
        return dict.find(x) != dict.end();
    }


    //@formatter:off
    template<typename T>
    std::vector<rtree::Item<T>> test_input_data(){
         auto boxes =  std::vector<mbr::MBR<T>>{
                {0, 0, 0, 0}, {10, 10, 10, 10}, {20, 20, 20, 20}, {25, 0, 25, 0}, {35, 10, 35, 10}, {45, 20, 45, 20}, {0, 25, 0, 25}, {10, 35, 10, 35},
                {20, 45, 20, 45}, {25, 25, 25, 25}, {35, 35, 35, 35}, {45, 45, 45, 45}, {50, 0, 50, 0}, {60, 10, 60, 10}, {70, 20, 70, 20}, {75, 0, 75, 0},
                {85, 10, 85, 10}, {95, 20, 95, 20}, {50, 25, 50, 25}, {60, 35, 60, 35}, {70, 45, 70, 45}, {75, 25, 75, 25}, {85, 35, 85, 35}, {95, 45, 95, 45},
                {0, 50, 0, 50}, {10, 60, 10, 60}, {20, 70, 20, 70}, {25, 50, 25, 50}, {35, 60, 35, 60}, {45, 70, 45, 70}, {0, 75, 0, 75}, {10, 85, 10, 85},
                {20, 95, 20, 95}, {25, 75, 25, 75}, {35, 85, 35, 85}, {45, 95, 45, 95}, {50, 50, 50, 50}, {60, 60, 60, 60}, {70, 70, 70, 70}, {75, 50, 75, 50},
                {85, 60, 85, 60}, {95, 70, 95, 70}, {50, 75, 50, 75}, {60, 85, 60, 85}, {70, 95, 70, 95}, {75, 75, 75, 75}, {85, 85, 85, 85}, {95, 95, 95, 95}
         };
         size_t id{0};
         std::vector<rtree::Item<T>> items ; 
         for (auto & box : boxes){
             items.emplace_back(rtree::Item{id++ , box});
         }
         return items; 
    }

    template<typename T>
    std::vector<rtree::Item<T>> test_input_empty_data(){
         auto infinity  = std::numeric_limits<double>::infinity();
         auto boxes =  std::vector<mbr::MBR<T>>{
            {-infinity, -infinity, infinity, infinity},
            {-infinity, -infinity, infinity, infinity},
            {-infinity, -infinity, infinity, infinity},
            {-infinity, -infinity, infinity, infinity},
            {-infinity, -infinity, infinity, infinity},
            {-infinity, -infinity, infinity, infinity},
         };
         size_t id{0};
         std::vector<rtree::Item<T>> items ;
         for (auto & box : boxes){
             items.emplace_back(rtree::Item{id++ , box});
         }
         return items;
    }
    //@formatter:on

    std::vector<rtree::Item<double>> someData(size_t n, rtree::Id id = 0) {
        //mbr::MBR<double>
        std::vector<rtree::Item<double>> data;
        data.reserve(n);
        for (size_t i = 0; i < n; i++) {
            data.emplace_back(
                    rtree::Item{
                            id++,
                            mbr::MBR{double(i), double(i), double(i), double(i)}
                    }
            );
        }

        return data;
    }

    template<typename T>
    void testResults(std::vector<rtree::Item<T>> nodes, std::vector<rtree::Item<T>> boxes, bool just_boxes = false) {
        std::sort(nodes.begin(), nodes.end());
        std::sort(boxes.begin(), boxes.end());

        REQUIRE(nodes.size() == boxes.size());
        if (just_boxes) {
            for (size_t i = 0; i < nodes.size(); i++) {
                if (!nodes[i].bbox().equals(boxes[i].bbox())) {
                    std::cout << "hey...\n";
                }
                REQUIRE(nodes[i].bbox().equals(boxes[i].bbox()));
            }
            return;
        }

        std::map<size_t, rtree::Item<T>> node_dict;
        for (auto &o: nodes) {
            if (has_key(node_dict, o.id)) {
                std::cout << o.id << '\n';
                std::cout << node_dict.at(o.id).box.wkt();
            }
            REQUIRE(!has_key(node_dict, o.id));
            node_dict[o.id] = o;
        }

        for (auto &o: boxes) {
            REQUIRE(has_key(node_dict, o.id));
            auto node_item = node_dict.at(o.id);
            REQUIRE(node_item.bbox().equals(o.bbox()));
        }
    }

    template<typename T>
    bool nodeEquals(rtree::Node<T> *a, rtree::Node<T> *b) {
        auto bln = a->bbox.equals(b->bbox);
        if (a->item.id != rtree::null_id && b->item.id != rtree::null_id) {
            bln = bln && a->item.bbox().equals(b->item.bbox());
        }

        bln = bln &&
              a->height == b->height &&
              a->leaf == b->leaf &&
              a->bbox == b->bbox &&
              a->children.size() == b->children.size();

        if (bln && !a->children.empty()) {
            for (size_t i = 0; bln && i < a->children.size(); i++) {
                bln = bln && nodeEquals(&a->children[i], &b->children[i]);
            }
        }

        return bln;
    }


    struct Pnt {
        double x;
        double y;

        mbr::MBR<double> bbox() {
            return mbr::MBR<double>{x, y, x + 2, y + 2};
        }

        bool operator==(const Pnt &other) {
            return x == other.x && y == other.y;
        }
    };

    // auto BenchData = GenDataItems(N, 1);
    // auto bboxes100 = GenDataItems(1000, 100*std::sqrt(0.1));
    // auto bboxes10 = GenDataItems(1000, 10);
    // auto bboxes1 = GenDataItems(1000, 1);
    // auto tree = NewRTree(maxFill).load(BenchData);
    template<typename T>
    std::vector<rtree::Item<T>> init_knn() {
        //@formatter:off
        std::vector<mbr::MBR<T>>  knn_data = {
            {87, 55, 87, 56}, {38, 13, 39, 16}, {7, 47, 8, 47}, {89, 9, 91, 12}, {4, 58, 5, 60}, {0, 11, 1, 12}, {0, 5, 0, 6}, {69, 78, 73, 78},
            {56, 77, 57, 81}, {23, 7, 24, 9}, {68, 24, 70, 26}, {31, 47, 33, 50}, {11, 13, 14, 15}, {1, 80, 1, 80}, {72, 90, 72, 91}, {59, 79, 61, 83},
            {98, 77, 101, 77}, {11, 55, 14, 56}, {98, 4, 100, 6}, {21, 54, 23, 58}, {44, 74, 48, 74}, {70, 57, 70, 61}, {32, 9, 33, 12}, {43, 87, 44, 91},
            {38, 60, 38, 60}, {62, 48, 66, 50}, {16, 87, 19, 91}, {5, 98, 9, 99}, {9, 89, 10, 90}, {89, 2, 92, 6}, {41, 95, 45, 98}, {57, 36, 61, 40},
            {50, 1, 52, 1}, {93, 87, 96, 88}, {29, 42, 33, 42}, {34, 43, 36, 44}, {41, 64, 42, 65}, {87, 3, 88, 4}, {56, 50, 56, 52}, {32, 13, 35, 15},
            {3, 8, 5, 11}, {16, 33, 18, 33}, {35, 39, 38, 40}, {74, 54, 78, 56}, {92, 87, 95, 90}, {12, 97, 16, 98}, {76, 39, 78, 40}, {16, 93, 18, 95},
            {62, 40, 64, 42}, {71, 87, 71, 88}, {60, 85, 63, 86}, {39, 52, 39, 56}, {15, 18, 19, 18}, {91, 62, 94, 63}, {10, 16, 10, 18}, {5, 86, 8, 87},
            {85, 85, 88, 86}, {44, 84, 44, 88}, {3, 94, 3, 97}, {79, 74, 81, 78}, {21, 63, 24, 66}, {16, 22, 16, 22}, {68, 97, 72, 97}, {39, 65, 42, 65},
            {51, 68, 52, 69}, {61, 38, 61, 42}, {31, 65, 31, 65}, {16, 6, 19, 6}, {66, 39, 66, 41}, {57, 32, 59, 35}, {54, 80, 58, 84}, {5, 67, 7, 71},
            {49, 96, 51, 98}, {29, 45, 31, 47}, {31, 72, 33, 74}, {94, 25, 95, 26}, {14, 7, 18, 8}, {29, 0, 31, 1}, {48, 38, 48, 40}, {34, 29, 34, 32},
            {99, 21, 100, 25}, {79, 3, 79, 4}, {87, 1, 87, 5}, {9, 77, 9, 81}, {23, 25, 25, 29}, {83, 48, 86, 51}, {79, 94, 79, 95}, {33, 95, 33, 99},
            {1, 14, 1, 14}, {33, 77, 34, 77}, {94, 56, 98, 59}, {75, 25, 78, 26}, {17, 73, 20, 74}, {11, 3, 12, 4}, {45, 12, 47, 12}, {38, 39, 39, 39},
            {99, 3, 103, 5}, {41, 92, 44, 96}, {79, 40, 79, 41}, {29, 2, 29, 4},
        };
        //@formatter:on

        size_t id{0};
        std::vector<rtree::Item<T>> data;
        for (auto &box: knn_data) {
            data.emplace_back(rtree::Item{id++, box});
        }
        return data;
    }

    bool found_in(const mbr::MBR<double> &needle, const std::vector<mbr::MBR<double>> &haystack) {
        auto found = false;
        for (auto &hay: haystack) {
            found = needle.equals(hay);
            if (found) {
                break;
            }
        }
        return found;
    }

    struct RichData {
        mbr::MBR<double> box = rtree::empty_mbr<double>();
        int version{-1};

        explicit RichData(mbr::MBR<double> _box, int _version) : box(_box), version(_version) {
        }

        mbr::MBR<double> &bbox() {
            return box.bbox();
        }
    };

    std::vector<RichData> fn_rich_data() {
        std::vector<RichData> richData;
        std::vector<mbr::MBR<double>> data = {
                {1,   2, 1,   2},
                {3,   3, 3,   3},
                {5,   5, 5,   5},
                {4,   2, 4,   2},
                {2,   4, 2,   4},
                {5,   3, 5,   3},
                {3,   4, 3,   4},
                {2.5, 4, 2.5, 4},
        };
        for (size_t i = 0; i < data.size(); i++) {
            richData.emplace_back(RichData(data[i], int(i) + 1));
        }
        return richData;
    }

    struct Parent {
        std::string wkt{};
        std::vector<std::string> children{};
    };

    template<typename U>
    std::vector<Parent> print_RTree(std::unique_ptr<rtree::Node<U>> &a) {
        std::vector<Parent> tokens;
        if (a == nullptr) {
            return tokens;
        }

        std::vector<rtree::Node<U> *> stack;
        stack.reserve(a->children.size());
        stack.emplace_back(a.get());
        while (!stack.empty()) {
            auto node = stack.back();
            stack.pop_back();
            auto parent = Parent{node->bbox.wkt()};
            //adopt children on stack and let node go out of scope
            for (auto &o: node->children) {
                auto n = o.get();
                if (!n->children.empty()) {
                    stack.emplace_back(n);
                    parent.children.emplace_back(n->bbox.wkt());
                }
            }
            if (!parent.children.empty()) {
                tokens.emplace_back(parent);
            }
        }
        return tokens;
    }
}

TEST_CASE("rtree 1", "[rtree 1]") {

    using namespace rtest;
    using namespace rtree;

    SECTION("should test load 9 & 10") {
        auto data = someData(0);
        auto tree0 = rtree::NewRTree<double>(0).load(data);
        REQUIRE(tree0.data.height == 1);

        auto data2 = someData(9);
        auto tree1 = NewRTree<double>(9).load(data2);
        REQUIRE(tree1.data.height == 1);

        auto data3 = someData(10);
        auto tree2 = NewRTree<double>(9).load(data3);
        REQUIRE(tree2.data.height == 2);
    }

    SECTION("tests search with some other") {
        std::vector<rtree::Item<double>> data{
                {1, {-115, 45,  -105, 55}},
                {2, {105,  45,  115,  55}},
                {3, {105,  -55, 115,  -45}},
                {4, {-115, -55, -105, -45}},
        };
        auto tree = NewRTree<double>(4);
        tree.load(data);
        auto res = tree.search({-180, -90, 180, 90});

        testResults(std::move(res), std::vector<rtree::Item<double>>{
                {1, {-115, 45,  -105, 55}},
                {2, {105,  45,  115,  55}},
                {3, {105,  -55, 115,  -45}},
                {4, {-115, -55, -105, -45}},
        }, true);

        testResults(tree.search(mbr::MBR<double>(-180, -90, 0, 90)), std::vector<rtree::Item<double>>{
                {1, {-115, 45,  -105, 55}},
                {2, {-115, -55, -105, -45}},
        }, true);

        testResults(tree.search(mbr::MBR<double>(0, -90, 180, 90)), std::vector<rtree::Item<double>>{
                {1, {105, 45,  115, 55}},
                {2, {105, -55, 115, -45}},
        }, true);

        testResults(tree.search(mbr::MBR<double>(-180, 0, 180, 90)), std::vector<rtree::Item<double>>{
                {1, {-115, 45, -105, 55}},
                {2, {105,  45, 115,  55}},
        }, true);

        testResults(tree.search(mbr::MBR<double>(-180, -90, 180, 0)), std::vector<rtree::Item<double>>{
                {1, {105,  -55, 115,  -45}},
                {2, {-115, -55, -105, -45}},
        }, true);
    }

    SECTION("#load uses standard insertion when given a low number of items") {
        auto data = rtest::test_input_data<double>();
        auto subslice = rtree::slice(data, 0, 3);
        auto rt = NewRTree<double>(8);
        rt.load(data);
        rt.load(subslice);

        auto tree = std::move(rt);

        auto data2 = rtest::test_input_data<double>();
        auto tree2 = NewRTree<double>(8);
        tree2.load(data2);
        tree2.insert(data2[0]);
        tree2.insert(data2[1]);
        tree2.insert(data2[2]);
        REQUIRE(nodeEquals(&tree.data, &tree2.data));
    }

    SECTION(" [int] #load uses standard insertion when given a low number of items") {
        auto data = rtest::test_input_data<int>();
        auto subslice = rtree::slice(data, 0, 3);
        rtree::RTree rt = NewRTree<int>(8);
        rt.load(data);
        rt.load(subslice);

        auto tree = std::move(rt);

        auto data2 = rtest::test_input_data<int>();
        rtree::RTree tree2 = NewRTree<int>(8);
        tree2.load(data2);
        tree2.insert(data2[0]);
        tree2.insert(data2[1]);
        tree2.insert(data2[2]);
        REQUIRE(nodeEquals(&tree.data, &tree2.data));
    }


    SECTION("[size_t] #load does nothing if (loading empty data)") {
        std::vector<rtree::Item<size_t>> data{};
        auto tree = NewRTree<size_t>(0);
        tree.load(data);
        REQUIRE(tree.is_empty());
        auto box = rtree::Item{0, mbr::MBR<size_t>{105, 45, 115, 55}};
        tree.insert(box);
        REQUIRE(!tree.is_empty());
    }


    SECTION("#load does nothing if (loading empty data)") {
        std::vector<rtree::Item<double>> data{};
        auto tree = NewRTree<double>(0);
        tree.load(data);
        REQUIRE(tree.is_empty());
    }

    SECTION("#load handles the insertion of maxEntries + 2 empty bboxes") {
        auto emptyData = test_input_empty_data<double>();
        auto tree = NewRTree<double>(4);
        tree.load(emptyData);
        REQUIRE(tree.data.height == 2);
        testResults(tree.all(), emptyData);
    }

    SECTION("#load handles the insertion of maxEntries + 2 empty bboxes") {
        auto emptyData = test_input_empty_data<double>();
        auto tree = NewRTree<double>(4);
        for (auto & i : emptyData) {
            tree.insert(i);
        }
        REQUIRE(tree.data.height == 2);
        testResults(tree.all(), emptyData);
        REQUIRE(tree.data.children[0].children.size() == 4);
        REQUIRE(tree.data.children[1].children.size() == 2);
    }

    SECTION("#load properly splits tree root when merging trees of the same height") {
        auto data = rtest::test_input_data<double>();
        std::vector<rtree::Item<double>> cloneData(data.begin(), data.end());
        std::vector<rtree::Item<double>> _cloneData(data.begin(), data.end());
        cloneData.insert(cloneData.end(), _cloneData.begin(), _cloneData.end());
        auto tree = NewRTree<double>(4);
        tree.load(data);
        tree.load(data);
        testResults(tree.all(), cloneData, true);
    }

    SECTION("#load properly merges data of smaller or bigger tree heights") {
        auto data = rtest::test_input_data<double>();
        auto smaller = someData(10, data.back().id + 100);

        std::vector<rtree::Item<double>> cloneData(data.begin(), data.end());
        cloneData.insert(cloneData.end(), smaller.begin(), smaller.end());

        auto tree1 = NewRTree<double>(4);
        tree1.load(data);
        tree1.load(smaller);
        auto tree2 = NewRTree<double>(4);
        tree2.load(smaller);
        tree2.load(data);
        REQUIRE(tree1.data.height == tree2.data.height);
        testResults(tree1.all(), cloneData);
        testResults(tree2.all(), cloneData);
    }

    SECTION("#load properly merges data of smaller or bigger tree heights 2") {
        auto N = static_cast<size_t > (8020);
        std::vector<rtree::Item<double>> smaller = GenDataItems(N, 1, 0);
        std::vector<rtree::Item<double>> larger = GenDataItems(2 * N, 1, smaller.back().id + 100);

        std::vector<rtree::Item<double>> cloneData(larger.begin(), larger.end());
        cloneData.insert(cloneData.end(), smaller.begin(), smaller.end());

        auto tree1 = NewRTree<double>(4);
        tree1.load(larger);
        tree1.load(smaller);

        auto tree2 = NewRTree<double>(4);
        tree2.load(smaller);
        tree2.load(larger);

        REQUIRE(tree1.data.height == tree2.data.height);
        testResults(tree1.all(), cloneData);
        testResults(tree2.all(), cloneData);

    }

    SECTION("#search finds matching points in the tree given a bbox") {
        auto data = rtest::test_input_data<double>();
        auto tree = NewRTree<double>(4);
        tree.load(data);
        //@formatter:off
        testResults(tree.search(mbr::MBR<double>(40, 20, 80, 70)), std::vector<rtree::Item<double>>{
                {1,{70, 20, 70, 20}}, {5,{75, 25, 75, 25}}, {9,{45, 45, 45, 45}},
                {2,{50, 50, 50, 50}}, {6,{60, 60, 60, 60}}, {10,{70, 70, 70, 70}},
                {3,{45, 20, 45, 20}}, {7,{45, 70, 45, 70}}, {11,{75, 50, 75, 50}},
                {4,{50, 25, 50, 25}}, {8,{60, 35, 60, 35}}, {12,{70, 45, 70, 45}},
        }, true);
        //@formatter:on
    }


    SECTION("#collides returns true when search finds matching points") {
        auto data = rtest::test_input_data<double>();
        auto tree = NewRTree<double>(4);
        tree.load(data);
        REQUIRE(tree.collides(mbr::MBR<double>(40, 20, 80, 70)));
        REQUIRE(!tree.collides(mbr::MBR<double>(200, 200, 210, 210)));
    }

    SECTION("#search returns an empty array if (nothing found") {
        auto data = rtest::test_input_data<double>();
        auto rt = NewRTree<double>(4);
        rt.load(data);
        auto result = rt.search(
                mbr::MBR<double>(200, 200, 210, 210)
        );
        REQUIRE(result.empty());
    }

    SECTION("#all <==>.Data returns all points in the tree") {
        auto data = rtest::test_input_data<double>();
        std::vector<rtree::Item<double>> cloneData(data.begin(), data.end());
        auto tree = NewRTree<double>(4);
        tree.load(data);
        auto result = tree.search(mbr::MBR<double>(0, 0, 100, 100));
        testResults(result, cloneData);
    }

    SECTION("#all <==>.Data returns all points in the tree") {
        std::vector<rtree::Item<double>> data = {
                {1, {0, 0, 0, 0}},
                {2, {2, 2, 2, 2}},
                {3, {1, 1, 1, 1}},
        };
        auto tree = NewRTree<double>(4);
        tree.load(data);
        auto box3333 = rtree::Item<double>{3333, {3, 3, 3, 3}};
        tree.insert(box3333);
        REQUIRE(tree.data.leaf);
        REQUIRE(tree.data.height == 1);
        REQUIRE(tree.data.bbox.equals(mbr::MBR<double>{0, 0, 3, 3}));
        std::vector<rtree::Item<double>> expects{
                {1,    {0, 0, 0, 0}},
                {2,    {2, 2, 2, 2}},
                {3,    {1, 1, 1, 1}},
                {3333, {3, 3, 3, 3}}
        };
        REQUIRE(tree.data.children.size() == expects.size());
        testResults(
                [&] {
                    std::vector<rtree::Item<double>> items;
                    for (auto &i: tree.data.children) {
                        items.emplace_back(i.item);
                    }
                    return items;
                }(),
                expects
        );
    }

    SECTION("[int]#insert forms a valid tree if (items are inserted one by one") {
        auto data = rtest::test_input_data<int>();
        auto tree = NewRTree<int>(4);

        for (auto &o: data) {
            tree.insert(o);
        }

        auto tree2 = NewRTree<int>(4);
        tree2.load(data);
        REQUIRE(tree.data.height - tree2.data.height <= 1);
        testResults(tree.all(), tree2.all());
    }

    SECTION("#insert forms a valid tree if (items are inserted one by one") {
        auto data = rtest::test_input_data<double>();
        auto tree = NewRTree<double>(4);

        for (auto &o: data) {
            tree.insert(o);
        }

        auto tree2 = NewRTree<double>(4);
        tree2.load(data);
        REQUIRE(tree.data.height - tree2.data.height <= 1);
        testResults(tree.all(), tree2.all());
    }

    SECTION("[size_t]#remove removes items correctly") {
        auto data = rtest::test_input_data<int>();
        auto N = len(data);
        std::vector<rtree::Item<int>> boxes;
        boxes.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            boxes.emplace_back(data[i]);
        }
        auto tree = NewRTree<int>(4);
        tree.load(boxes);
        tree.remove(data[0]).remove(data[1]).remove(data[2]);
        tree.remove(boxes[N - 1]).remove(boxes[N - 2]).remove(boxes[N - 3]);

        std::vector<rtree::Item<int>> cloneData(data.begin() + 3, data.end() - 3);
        testResults(tree.all(), cloneData);
    }

    SECTION("#remove removes items correctly") {
        auto data = rtest::test_input_data<double>();
        auto N = len(data);
        std::vector<rtree::Item<double>> items{};
        items.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            items.emplace_back(data[i]);
        }
        auto tree = NewRTree<double>(4);
        tree.load(items);
        tree.remove(data[0]).remove(data[1]).remove(data[2]);

        tree.remove(items[N - 1]);
        tree.remove(items[N - 2]);
        tree.remove(items[N - 3]);

        std::vector<rtree::Item<double>> cloneData(data.begin() + 3, data.end() - 3);
        testResults(tree.all(), cloneData);
    }

    SECTION("#remove does nothing if (nothing found)") {
        rtree::Item<double> item{};
        auto data = rtest::test_input_data<double>();

        auto tree = NewRTree<double>(0);
        tree.load(data);

        auto data2 = rtest::test_input_data<double>();
        auto tree2 = NewRTree<double>(0);
        tree2.load(data2);

        tree2.remove(mbr::MBR<double>(13, 13, 13, 13));
        REQUIRE(nodeEquals(&tree.data, &tree2.data));

        tree2.remove(item); //not init
        REQUIRE(nodeEquals(&tree.data, &tree2.data));
    }

    SECTION("#remove brings the tree to a clear state when removing everything one by one") {
        auto data = rtest::test_input_data<double>();
        auto tree = NewRTree<double>(4);
        tree.load(data);
        auto result = tree.search(mbr::MBR<double>(0, 0, 100, 100));

        for (size_t i = 0; i < len(result); i++) {
            tree.remove(result[i]);
        }
        auto result2 = tree.search(mbr::MBR<double>(0, 0, 100, 100));
        REQUIRE(tree.is_empty());
    }

    SECTION("#clear should clear all the data in the tree") {
        auto data = rtest::test_input_data<double>();
        auto tree = NewRTree<double>(4);
        tree.load(data);
        tree.clear();
        REQUIRE(tree.is_empty());
    }

    SECTION("should have chainable API") {
        auto data = rtest::test_input_data<double>();
        auto rt = NewRTree<double>(4);
        rt.load(data);
        rt.insert(data[0]);
        rt.remove(data[0]);
        rt.clear();
        REQUIRE(rt.is_empty());
    }
}

TEST_CASE("rtree 2", "[rtree util]") {
    using namespace rtree;
    using namespace rtest;

    SECTION("tests pop nodes") {
        auto abox = rtree::Item<double>{};
        auto bbox = rtree::Item<double>{};
        auto cbox = rtree::Item<double>{};
        auto a = NewNode<double>(abox, 0, true, std::vector<Node<double>>{});
        auto b = NewNode<double>(bbox, 1, true, std::vector<Node<double>>{});
        auto c = NewNode<double>(cbox, 1, true, std::vector<Node<double>>{});
        std::vector<Node<double> *> nodes;
        Node<double> *n;

        n = pop(nodes);
        REQUIRE(n == nullptr);

        //nodes = {a, b, c};
        nodes.emplace_back(&a);
        nodes.emplace_back(&b);
        nodes.emplace_back(&c);
        REQUIRE(len(nodes));

        n = pop(nodes);
        REQUIRE(len(nodes) == 2);
        REQUIRE(n == &c);

        n = pop(nodes);
        REQUIRE(len(nodes) == 1);
        REQUIRE(n == &b);

        n = pop(nodes);
        REQUIRE(len(nodes) == 0);
        REQUIRE(n == &a);

        n = pop(nodes);
        REQUIRE(len(nodes) == 0);
        REQUIRE(n == nullptr);
    }

    SECTION("tests pop index") {
        size_t a = 0;
        size_t b = 1;
        size_t c = 2;
        std::vector<size_t> indexes{};
        size_t n = 0;

        REQUIRE(len(indexes) == 0);

        indexes = {a, b, c};
        REQUIRE(len(indexes) == 3);

        n = pop(indexes);
        REQUIRE(len(indexes) == 2);
        REQUIRE(n == c);

        n = pop(indexes);
        REQUIRE(len(indexes) == 1);
        REQUIRE(n == b);

        n = pop(indexes);
        REQUIRE(len(indexes) == 0);
        REQUIRE(n == a);
    }
}


TEST_CASE("rtree knn", "[rtree knn]") {

    using namespace rtree;
    using namespace rtest;

    SECTION("finds n neighbours") {
        auto knn_data = rtest::init_knn<double>();
        auto rt = NewRTree<double>(9);
        rt.load(knn_data);
        auto nn = rt.KNN(mbr::MBR<double>(40, 40, 40, 40), 10,
                         [](const mbr::MBR<double> &query, KObj<double> obj) {
                             return query.distance(obj.bbox);
                         });
        //@formatter:off
        std::vector<mbr::MBR<double>> result = {
                {38, 39, 39, 39}, {35, 39, 38, 40}, {34, 43, 36, 44}, {29, 42, 33, 42},
                {48, 38, 48, 40}, {31, 47, 33, 50}, {34, 29, 34, 32}, {29, 45, 31, 47},
                {39, 52, 39, 56}, {57, 36, 61, 40},
        };

        REQUIRE(len(nn) == len(result));
        //@formatter:on
        for (auto &n: nn) {
            REQUIRE(rtest::found_in(n.bbox(), result));
        }
    }

    SECTION("finds n neighbours with geoms") {
        auto knn_data = rtest::init_knn<double>();
        std::vector<mbr::MBR<double>> predicate_mbr;

        auto scoreFunc = [](const mbr::MBR<double> &query, KObj<double> obj) {
            return query.distance(obj.bbox);
        };

        auto createPredicate = [&](double dist) {
            return [=, &predicate_mbr](KObj<double> candidate) {
                REQUIRE(candidate.is_item);
                if (candidate.score() <= dist) {
                    predicate_mbr.emplace_back(candidate.bbox);
                    return std::tuple<bool, bool>(true, false);
                }
                return std::tuple<bool, bool>(false, true);
            };
        };
        auto rt = NewRTree<double>(9);
        rt.load(knn_data);
        auto prefFn = createPredicate(6);
        auto query = mbr::MBR<double>(
                74.88825108886668, 82.678427498132,
                74.88825108886668, 82.678427498132
        );

        auto res = rt.KNN(query, 10, scoreFunc, prefFn);

        REQUIRE(len(res) == 2);
        for (size_t i = 0; i < res.size(); i++) {
            REQUIRE(res[i].bbox() == predicate_mbr[i]);
        }
    }

    SECTION("find n neighbours that do satisfy a given predicate") {
        auto knn_data = rtest::init_knn<double>();
        auto rich_data = fn_rich_data();
        std::vector<rtree::Item<double>> objects;
        size_t id{0};;
        for (auto &d: rich_data) {
            objects.emplace_back(rtree::Item{id++, d.box});
        }
        auto rt = NewRTree<double>(9);
        rt.load(objects);

        auto scoreFn = [](mbr::MBR<double> query, KObj<double> boxer) {
            return query.distance(boxer.bbox);
        };

        auto predicate = [&](KObj<double> v) {
            auto &o = rich_data[v.get_item().id];
            return std::tuple<bool, bool>(o.version < 5, false);
        };

        auto result = rt.KNN(mbr::MBR<double>(2, 4, 2, 4), 1, scoreFn, predicate);

        REQUIRE(len(result) == 1);

        auto v = result[0];
        auto expects_mbr = mbr::MBR<double>{3, 3, 3, 3};
        auto expects_version = 2;

        REQUIRE(v.box == expects_mbr);
        REQUIRE(rich_data.at(v.id).version == expects_version);
    }
}

TEST_CASE("rtree build - bulkload", "[rtree build - bulkload]") {
    using namespace rtree;
    using namespace rtest;

    SECTION("same root bounds for : bulkload & single insert ") {
        //@formatter:off
        std::vector<mbr::MBR<double>> data = {
			{30.74124324842736, 1.5394264094726768, 35.574749911400275, 8.754917282902216}, {7.381378714281472, 64.86180480586492, 19.198256264240655, 68.0987794848029}, {55.08436657887449, 73.66959671568338, 64.23351406225139, 77.82561878388375}, {60.0196123955198, 57.30699966964475, 74.83877476995968, 71.6532691707469}, {70.41627091953383, 51.438036044803454, 80.79446144066551, 55.724409001469795}, {6.483303127937942, 80.37332301675087, 6.50914529921677, 82.02059482017387}, {46.67649373819957, 64.24510021830747, 49.2622050275365, 72.2146377872009}, {13.896809634528902, 52.75698091860803, 27.3474212705194, 59.708006858014954}, {45.352809515933565, 67.57279878792961, 57.71107486286911, 80.63410132702094}, {58.12911437270156, 21.853066059797676, 72.6816258699198, 25.407156729750344}, {1.228055380876119, 71.76243208229317, 3.921389356330319, 71.81985676158466}, {24.176338710683243, 40.468612774474124, 30.028427694218617, 54.92587462821439}, {75.90272549273205, 70.78950716967577, 90.24958662679839, 73.14532201100896},
			{81.17621599681077, 43.92908059235767, 90.4623706429688, 45.683200269169774}, {10.765947677699227, 81.39085907882142, 16.395569791144023, 89.08943214908905}, {54.98460948258535, 75.98770610541906, 63.17175560560506, 89.58032814388704}, {42.47968070466303, 70.33863394618999, 53.969718678982176, 81.12499083427267}, {56.597735425362224, 22.872881616226724, 58.02513594712652, 29.461626653458254}, {28.072656807817236, 3.648771707777917, 32.25507880635046, 14.896956422497794}, {49.07401457054004, 65.43509168217955, 50.276686480083214, 72.13126764274583}, {66.92950379018822, 7.40714495221543, 78.79495207418685, 15.349551257658238}, {70.05814537971477, 81.30351958853318, 71.64399891813584, 91.16708488214654}, {21.4511094375575, 69.72891964401825, 31.722373869482286, 80.3256272486247}, {40.232777196706415, 26.760849136982976, 52.202812069867704, 34.21206366219117}, {2.368032571076858, 16.296113490306034, 12.33270360370716, 30.694571126478845}, {9.01855144170366, 55.970132314222134, 23.827554767436514, 60.48030769802354}, {80.61271599954031, 36.74002124278151, 91.79275857224492, 46.9506194268175}, {50.34990344969663, 81.49769656350821, 63.617315842569894, 83.30755417296216}, {39.18113381327339, 62.28148778267892, 46.4815234729194, 67.41798018502531},
			{29.998914416747247, 11.59250655284693, 33.376874697786775, 12.379204853229147}, {81.64879583058361, 25.545401825528394, 93.4343371235951, 37.16442658088167}, {38.58905494531754, 31.87745905840195, 41.7616624289497, 38.45823126735888}, {0.9178278426197698, 24.298283582889418, 13.300394793306303, 29.32894041204992}, {65.26849055356847, 81.26949067385523, 69.4019309878049, 95.14982799740329}, {41.57395146960945, 42.58630560128803, 44.74131455539111, 52.67240067840212}, {78.75491794797742, 24.519635432090283, 86.62303951191035, 27.152009252646756}, {57.413508019097335, 16.222132563535784, 64.52460425468645, 26.468580365950785}, {38.70624110521209, 63.6483778012707, 42.81587531412866, 76.69707330624905}, {45.79681150909137, 40.50191132346466, 56.183424730475984, 45.059343488954596}, {59.12908726623217, 61.8670788267583, 72.67061706103317, 74.71825120772677}, {53.530204647536515, 22.210826106446316, 56.19567351522378, 36.70783763707212}, {66.56685327399163, 41.84620000931149, 67.95502218856858, 51.90145172377749}, {13.647425280602949, 48.287305203367325, 14.605520880072303, 50.785335362500966}, {9.580714642281816, 71.82612512759374, 22.052586035203777, 78.60447881685704}, {42.52476287398914, 31.798014129805892, 47.30017532169579, 43.32042676277269}, {15.231406548475704, 20.91813524362627, 27.999049905750184, 33.12719299053375}, {68.25622304622375, 36.45344418954924, 75.12753345668115, 42.96412962336906},
			{24.674565636296396, 61.64103736035227, 33.35950737775334, 68.17273669513995}, {27.860994552259186, 54.07784655778231, 37.454370732019164, 55.03748662118532}, {12.989350409059881, 12.850601894307912, 19.63701743062105, 24.447201165885136}, {54.351699198645946, 38.669663277102835, 62.70698234918281, 50.77799147478973}, {5.195541592506005, 27.378150434771385, 12.470640457055284, 31.42600927621769}, {50.42859019394414, 76.74400020764121, 61.43712226636309, 81.94545584300995}, {78.94947703076804, 80.53231036050055, 80.65894334965007, 80.53525709875574}, {25.444253736005553, 7.68730085456098, 31.065085510940172, 20.3498357552189}, {67.23805308545823, 13.569845282055715, 72.08492158784647, 28.386336312117162}, {73.53304448250748, 72.95399805919209, 78.88399497592506, 86.10583870486123}, {5.128991214483967, 46.433989769953975, 10.301559209436643, 47.47697754635162}, {34.345971501358505, 37.67046253655506, 46.65109226249595, 43.20929547370596}, {46.288476425780644, 83.24699351224912, 53.04617705157806, 95.25275555638714}, {2.3371253741744717, 67.38540121025542, 13.258004924360035, 67.9350571187773}, {81.50701949936798, 12.96213890790966, 90.69810567341676, 26.897004984394016}, {19.618219504752606, 35.07620582977229, 22.719692101944606, 35.682818900087824}, {12.212115116661117, 56.27156067476181, 15.934817779897248, 62.75070913000411}, {68.37555295280667, 52.219237356472945, 68.38823378366567, 63.48647393313754},
			{30.62554452606222, 60.101485548798514, 37.063824618295754, 71.04525498101337}, {56.032005794131614, 71.80616208209968, 67.22546752158931, 83.70215276205255}, {20.14317265947747, 73.77798886182363, 34.25432987619779, 87.24104072094495}, {10.507678860183212, 66.06446404977234, 22.91945017863563, 73.50576752587352}, {26.0796380640738, 39.08543029877627, 37.497243272316375, 42.198598580655705}, {58.204665266130036, 58.20119021138755, 66.86094220293387, 61.613651791527374}, {40.43959914994069, 2.5737454435527933, 47.14440867190218, 10.136829689608973}, {81.61166337839565, 57.04686555019882, 82.13024015743876, 60.52557802686094}, {1.1438702774984308, 64.4390551345789, 1.207827079116793, 74.94606495692364}, {22.698477311365394, 31.694032934311718, 23.012351437738243, 34.826851291697004}, {58.23302290469934, 63.09245797513119, 63.89603555830784, 71.13299682623365}, {1.1209075169457285, 81.28342384198416, 2.010664217814431, 85.39246047317187}, {12.031894943077951, 47.03188640891187, 17.157531829906453, 58.84050109551066}, {25.175447117884868, 53.84501614745653, 29.018643250506607, 59.38873449198591}, {2.2848309030370015, 13.908167333298184, 9.169561431787841, 19.16049137202979}, {50.013550661499245, 78.5109200392331, 61.27884750099618, 90.82242857844415},
			{60.35181123067779, 50.30720879159393, 66.40423614499642, 62.711248070454005}, {12.818633233242565, 80.69085735063159, 25.51374909020891, 93.22537975149076}, {13.89435574446365, 30.374627423660982, 26.014177608552792, 40.22893652344269}, {68.59949104329682, 71.57717815724429, 71.14413101711249, 81.32143731631942}, {8.759053910523154, 40.17136447593845, 22.076247428918848, 51.97034411093291}, {75.0237223114521, 10.812195153356786, 75.45859644475163, 24.680056123348074}, {37.640987086884465, 44.31736944555115, 46.79079124130418, 52.298119297002756}, {77.86465045295246, 69.74685405122065, 91.0727578759392, 81.32602647164121}, {41.571023531510896, 41.188931957868, 47.81613155473583, 53.78551712929363}, {46.21623238891625, 12.566288400974617, 60.42998852835609, 23.520076065312416}, {39.651498265328506, 13.503482197678323, 50.2456922936693, 17.970333385957133}, {22.002987425318885, 4.223514231931571, 24.39665459195155, 17.79996696134728}, {10.238509846935935, 17.775671898372956, 24.90139389081459, 30.900047607940877}, {11.945673076143192, 11.005643838128806, 14.458677679728162, 25.935774067123525}, {34.15254570484473, 32.9087837466544, 39.806374568647804, 45.792474254223166}, {1.2619249479259986, 73.38259039620652, 5.732709854315865, 82.08100065666045},
			{68.88687814624431, 70.06499982957165, 70.86758866753506, 78.39070584782843}, {53.346140703038856, 38.61621943306142, 58.18001677406793, 46.227279405415416}, {60.91283806646173, 5.328797186659199, 70.97382774644399, 11.165367727083606},
		};
      //@formatter:on

        size_t node_size = 0;
        auto oneT = NewRTree<double>(9);
        auto one_defT = NewRTree<double>(node_size);
        auto bulkT = NewRTree<double>(9);
        //one by one
        std::vector<rtree::Item<double>> data_oneByone{};
        for (size_t i = 0; i < data.size(); i++) {
            data_oneByone.emplace_back(rtree::Item{i, data[i]});
        }
        for (auto &d: data_oneByone) { oneT.insert(d); }

        //fill zero size
        for (auto &d: data_oneByone) { one_defT.insert(d); }

        auto one_mbr = oneT.data.bbox;
        auto one_def_mbr = one_defT.data.bbox;

        //bulkload
        std::vector<rtree::Item<double>> bulk_items;
        for (size_t i = 0; i < data.size(); i++) {
            bulk_items.emplace_back(rtree::Item{i + 222222, data[i]});
        }
        bulkT.load(bulk_items);
        auto buk_mbr = bulkT.data.bbox;

        REQUIRE(one_mbr.equals(one_def_mbr));
        REQUIRE(one_mbr.equals(buk_mbr));
        REQUIRE(std::abs(int(bulkT.data.height - oneT.data.height)) <= 1);
        REQUIRE(len(bulkT.data.children) == len(oneT.data.children));

//        auto tokens = print_RTree(oneT.data);
//        for (auto& tok : tokens) {
//            std::cout << tok.wkt << std::endl;
//            for (auto& ch: tok.children) {
//                std::cout << "    " << ch << std::endl;
//            }
//        }
    }

    SECTION("build rtree by and remove all") {
        //@formatter:off
            std::vector<mbr::MBR<double>> data ={
                {30.74124324842736, 1.5394264094726768, 35.574749911400275, 8.754917282902216}, {7.381378714281472, 64.86180480586492, 19.198256264240655, 68.0987794848029}, {55.08436657887449, 73.66959671568338, 64.23351406225139, 77.82561878388375}, {60.0196123955198, 57.30699966964475, 74.83877476995968, 71.6532691707469}, {70.41627091953383, 51.438036044803454, 80.79446144066551, 55.724409001469795}, {6.483303127937942, 80.37332301675087, 6.50914529921677, 82.02059482017387}, {46.67649373819957, 64.24510021830747, 49.2622050275365, 72.2146377872009}, {13.896809634528902, 52.75698091860803, 27.3474212705194, 59.708006858014954}, {45.352809515933565, 67.57279878792961, 57.71107486286911, 80.63410132702094}, {58.12911437270156, 21.853066059797676, 72.6816258699198, 25.407156729750344}, {1.228055380876119, 71.76243208229317, 3.921389356330319, 71.81985676158466}, {24.176338710683243, 40.468612774474124, 30.028427694218617, 54.92587462821439}, {75.90272549273205, 70.78950716967577, 90.24958662679839, 73.14532201100896},
                {81.17621599681077, 43.92908059235767, 90.4623706429688, 45.683200269169774}, {10.765947677699227, 81.39085907882142, 16.395569791144023, 89.08943214908905}, {54.98460948258535, 75.98770610541906, 63.17175560560506, 89.58032814388704}, {42.47968070466303, 70.33863394618999, 53.969718678982176, 81.12499083427267}, {56.597735425362224, 22.872881616226724, 58.02513594712652, 29.461626653458254}, {28.072656807817236, 3.648771707777917, 32.25507880635046, 14.896956422497794}, {49.07401457054004, 65.43509168217955, 50.276686480083214, 72.13126764274583}, {66.92950379018822, 7.40714495221543, 78.79495207418685, 15.349551257658238}, {70.05814537971477, 81.30351958853318, 71.64399891813584, 91.16708488214654}, {21.4511094375575, 69.72891964401825, 31.722373869482286, 80.3256272486247}, {40.232777196706415, 26.760849136982976, 52.202812069867704, 34.21206366219117}, {2.368032571076858, 16.296113490306034, 12.33270360370716, 30.694571126478845}, {9.01855144170366, 55.970132314222134, 23.827554767436514, 60.48030769802354}, {80.61271599954031, 36.74002124278151, 91.79275857224492, 46.9506194268175}, {50.34990344969663, 81.49769656350821, 63.617315842569894, 83.30755417296216}, {39.18113381327339, 62.28148778267892, 46.4815234729194, 67.41798018502531},
                {29.998914416747247, 11.59250655284693, 33.376874697786775, 12.379204853229147}, {81.64879583058361, 25.545401825528394, 93.4343371235951, 37.16442658088167}, {38.58905494531754, 31.87745905840195, 41.7616624289497, 38.45823126735888}, {0.9178278426197698, 24.298283582889418, 13.300394793306303, 29.32894041204992}, {65.26849055356847, 81.26949067385523, 69.4019309878049, 95.14982799740329}, {41.57395146960945, 42.58630560128803, 44.74131455539111, 52.67240067840212}, {78.75491794797742, 24.519635432090283, 86.62303951191035, 27.152009252646756}, {57.413508019097335, 16.222132563535784, 64.52460425468645, 26.468580365950785}, {38.70624110521209, 63.6483778012707, 42.81587531412866, 76.69707330624905}, {45.79681150909137, 40.50191132346466, 56.183424730475984, 45.059343488954596}, {59.12908726623217, 61.8670788267583, 72.67061706103317, 74.71825120772677}, {53.530204647536515, 22.210826106446316, 56.19567351522378, 36.70783763707212}, {66.56685327399163, 41.84620000931149, 67.95502218856858, 51.90145172377749}, {13.647425280602949, 48.287305203367325, 14.605520880072303, 50.785335362500966}, {9.580714642281816, 71.82612512759374, 22.052586035203777, 78.60447881685704}, {42.52476287398914, 31.798014129805892, 47.30017532169579, 43.32042676277269}, {15.231406548475704, 20.91813524362627, 27.999049905750184, 33.12719299053375}, {68.25622304622375, 36.45344418954924, 75.12753345668115, 42.96412962336906},
                {24.674565636296396, 61.64103736035227, 33.35950737775334, 68.17273669513995}, {27.860994552259186, 54.07784655778231, 37.454370732019164, 55.03748662118532}, {12.989350409059881, 12.850601894307912, 19.63701743062105, 24.447201165885136}, {54.351699198645946, 38.669663277102835, 62.70698234918281, 50.77799147478973}, {5.195541592506005, 27.378150434771385, 12.470640457055284, 31.42600927621769}, {50.42859019394414, 76.74400020764121, 61.43712226636309, 81.94545584300995}, {78.94947703076804, 80.53231036050055, 80.65894334965007, 80.53525709875574}, {25.444253736005553, 7.68730085456098, 31.065085510940172, 20.3498357552189}, {67.23805308545823, 13.569845282055715, 72.08492158784647, 28.386336312117162}, {73.53304448250748, 72.95399805919209, 78.88399497592506, 86.10583870486123}, {5.128991214483967, 46.433989769953975, 10.301559209436643, 47.47697754635162}, {34.345971501358505, 37.67046253655506, 46.65109226249595, 43.20929547370596}, {46.288476425780644, 83.24699351224912, 53.04617705157806, 95.25275555638714}, {2.3371253741744717, 67.38540121025542, 13.258004924360035, 67.9350571187773}, {81.50701949936798, 12.96213890790966, 90.69810567341676, 26.897004984394016}, {19.618219504752606, 35.07620582977229, 22.719692101944606, 35.682818900087824}, {12.212115116661117, 56.27156067476181, 15.934817779897248, 62.75070913000411}, {68.37555295280667, 52.219237356472945, 68.38823378366567, 63.48647393313754},
                {30.62554452606222, 60.101485548798514, 37.063824618295754, 71.04525498101337}, {56.032005794131614, 71.80616208209968, 67.22546752158931, 83.70215276205255}, {20.14317265947747, 73.77798886182363, 34.25432987619779, 87.24104072094495}, {10.507678860183212, 66.06446404977234, 22.91945017863563, 73.50576752587352}, {26.0796380640738, 39.08543029877627, 37.497243272316375, 42.198598580655705}, {58.204665266130036, 58.20119021138755, 66.86094220293387, 61.613651791527374}, {40.43959914994069, 2.5737454435527933, 47.14440867190218, 10.136829689608973}, {81.61166337839565, 57.04686555019882, 82.13024015743876, 60.52557802686094}, {1.1438702774984308, 64.4390551345789, 1.207827079116793, 74.94606495692364}, {22.698477311365394, 31.694032934311718, 23.012351437738243, 34.826851291697004}, {58.23302290469934, 63.09245797513119, 63.89603555830784, 71.13299682623365}, {1.1209075169457285, 81.28342384198416, 2.010664217814431, 85.39246047317187}, {12.031894943077951, 47.03188640891187, 17.157531829906453, 58.84050109551066}, {25.175447117884868, 53.84501614745653, 29.018643250506607, 59.38873449198591}, {2.2848309030370015, 13.908167333298184, 9.169561431787841, 19.16049137202979}, {50.013550661499245, 78.5109200392331, 61.27884750099618, 90.82242857844415},
                {60.35181123067779, 50.30720879159393, 66.40423614499642, 62.711248070454005}, {12.818633233242565, 80.69085735063159, 25.51374909020891, 93.22537975149076}, {13.89435574446365, 30.374627423660982, 26.014177608552792, 40.22893652344269}, {68.59949104329682, 71.57717815724429, 71.14413101711249, 81.32143731631942}, {8.759053910523154, 40.17136447593845, 22.076247428918848, 51.97034411093291}, {75.0237223114521, 10.812195153356786, 75.45859644475163, 24.680056123348074}, {37.640987086884465, 44.31736944555115, 46.79079124130418, 52.298119297002756}, {77.86465045295246, 69.74685405122065, 91.0727578759392, 81.32602647164121}, {41.571023531510896, 41.188931957868, 47.81613155473583, 53.78551712929363}, {46.21623238891625, 12.566288400974617, 60.42998852835609, 23.520076065312416}, {39.651498265328506, 13.503482197678323, 50.2456922936693, 17.970333385957133}, {22.002987425318885, 4.223514231931571, 24.39665459195155, 17.79996696134728}, {10.238509846935935, 17.775671898372956, 24.90139389081459, 30.900047607940877}, {11.945673076143192, 11.005643838128806, 14.458677679728162, 25.935774067123525}, {34.15254570484473, 32.9087837466544, 39.806374568647804, 45.792474254223166}, {1.2619249479259986, 73.38259039620652, 5.732709854315865, 82.08100065666045},
                {68.88687814624431, 70.06499982957165, 70.86758866753506, 78.39070584782843}, {53.346140703038856, 38.61621943306142, 58.18001677406793, 46.227279405415416}, {60.91283806646173, 5.328797186659199, 70.97382774644399, 11.165367727083606},
            };
            //@formatter:on
        mbr::MBR<double> query = {0., 0., 100, 100};

        auto tree = NewRTree<double>(0);
        std::vector<rtree::Item<double>> data_oneByone{};
        for (size_t i = 0; i < data.size(); i++) {
            data_oneByone.emplace_back(rtree::Item{i, data[i]});
        }

        auto res = tree.search(query);
        for (auto &o: res) {
            tree.remove(o);
        }
        REQUIRE(tree.is_empty());
        REQUIRE(tree.data.children.empty());
        REQUIRE(tree.data.bbox == empty_mbr<double>());
    }

    SECTION("search for items in tree") {
        //@formatter:off
        std::vector<mbr::MBR<double>> data = {
                {30.74124324842736,  1.5394264094726768, 35.574749911400275, 8.754917282902216}, {7.381378714281472,  64.86180480586492,  19.198256264240655, 68.0987794848029}, {55.08436657887449,  73.66959671568338,  64.23351406225139,  77.82561878388375}, {60.0196123955198,   57.30699966964475,  74.83877476995968,  71.6532691707469}, {70.41627091953383,  51.438036044803454, 80.79446144066551,  55.724409001469795}, {6.483303127937942,  80.37332301675087,  6.50914529921677,   82.02059482017387}, {46.67649373819957,  64.24510021830747,  49.2622050275365,   72.2146377872009}, {13.896809634528902, 52.75698091860803,  27.3474212705194,   59.708006858014954}, {45.352809515933565, 67.57279878792961,  57.71107486286911,  80.63410132702094}, {58.12911437270156,  21.853066059797676, 72.6816258699198,   25.407156729750344}, {1.228055380876119,  71.76243208229317,  3.921389356330319,  71.81985676158466}, {24.176338710683243, 40.468612774474124, 30.028427694218617, 54.92587462821439}, {75.90272549273205,  70.78950716967577,  90.24958662679839,  73.14532201100896}, {81.17621599681077,  43.92908059235767,  90.4623706429688,   45.683200269169774}, {10.765947677699227, 81.39085907882142,  16.395569791144023, 89.08943214908905}, {54.98460948258535,  75.98770610541906,  63.17175560560506,  89.58032814388704}, {42.47968070466303,  70.33863394618999,  53.969718678982176, 81.12499083427267}, {56.597735425362224, 22.872881616226724, 58.02513594712652,  29.461626653458254}, {28.072656807817236, 3.648771707777917,  32.25507880635046,  14.896956422497794}, {49.07401457054004,  65.43509168217955,  50.276686480083214, 72.13126764274583}, {66.92950379018822,  7.40714495221543,   78.79495207418685,  15.349551257658238}, {70.05814537971477,  81.30351958853318,  71.64399891813584,  91.16708488214654}, {21.4511094375575,   69.72891964401825,  31.722373869482286, 80.3256272486247}, {40.232777196706415, 26.760849136982976, 52.202812069867704, 34.21206366219117}, {2.368032571076858,  16.296113490306034, 12.33270360370716,  30.694571126478845}, {9.01855144170366,   55.970132314222134, 23.827554767436514, 60.48030769802354}, {80.61271599954031,  36.74002124278151,  91.79275857224492,  46.9506194268175}, {50.34990344969663,  81.49769656350821,  63.617315842569894, 83.30755417296216}, {39.18113381327339,  62.28148778267892,  46.4815234729194,   67.41798018502531}, {29.998914416747247, 11.59250655284693,  33.376874697786775, 12.379204853229147}, {81.64879583058361,  25.545401825528394, 93.4343371235951,   37.16442658088167}, {38.58905494531754,  31.87745905840195,  41.7616624289497,   38.45823126735888}, {0.9178278426197698, 24.298283582889418, 13.300394793306303, 29.32894041204992}, {65.26849055356847,  81.26949067385523,  69.4019309878049,   95.14982799740329}, {41.57395146960945,  42.58630560128803,  44.74131455539111,  52.67240067840212}, {78.75491794797742,  24.519635432090283, 86.62303951191035,  27.152009252646756}, {57.413508019097335, 16.222132563535784, 64.52460425468645,  26.468580365950785}, {38.70624110521209,  63.6483778012707,   42.81587531412866,  76.69707330624905}, {45.79681150909137,  40.50191132346466,  56.183424730475984, 45.059343488954596}, {59.12908726623217,  61.8670788267583,   72.67061706103317,  74.71825120772677}, {53.530204647536515, 22.210826106446316, 56.19567351522378,  36.70783763707212}, {66.56685327399163,  41.84620000931149,  67.95502218856858,  51.90145172377749}, {13.647425280602949, 48.287305203367325, 14.605520880072303, 50.785335362500966}, {9.580714642281816,  71.82612512759374,  22.052586035203777, 78.60447881685704}, {42.52476287398914,  31.798014129805892, 47.30017532169579,  43.32042676277269}, {15.231406548475704, 20.91813524362627,  27.999049905750184, 33.12719299053375}, {68.25622304622375,  36.45344418954924,  75.12753345668115,  42.96412962336906}, {24.674565636296396, 61.64103736035227,  33.35950737775334,  68.17273669513995}, {27.860994552259186, 54.07784655778231,  37.454370732019164, 55.03748662118532}, {12.989350409059881, 12.850601894307912, 19.63701743062105,  24.447201165885136}, {54.351699198645946, 38.669663277102835, 62.70698234918281,  50.77799147478973}, {5.195541592506005,  27.378150434771385, 12.470640457055284, 31.42600927621769}, {50.42859019394414,  76.74400020764121,  61.43712226636309,  81.94545584300995}, {78.94947703076804,  80.53231036050055,  80.65894334965007,  80.53525709875574}, {25.444253736005553, 7.68730085456098,   31.065085510940172, 20.3498357552189}, {67.23805308545823,  13.569845282055715, 72.08492158784647,  28.386336312117162}, {73.53304448250748,  72.95399805919209,  78.88399497592506,  86.10583870486123}, {5.128991214483967,  46.433989769953975, 10.301559209436643, 47.47697754635162}, {34.345971501358505, 37.67046253655506,  46.65109226249595,  43.20929547370596}, {46.288476425780644, 83.24699351224912,  53.04617705157806,  95.25275555638714}, {2.3371253741744717, 67.38540121025542,  13.258004924360035, 67.9350571187773}, {81.50701949936798,  12.96213890790966,  90.69810567341676,  26.897004984394016}, {19.618219504752606, 35.07620582977229,  22.719692101944606, 35.682818900087824}, {12.212115116661117, 56.27156067476181,  15.934817779897248, 62.75070913000411}, {68.37555295280667,  52.219237356472945, 68.38823378366567,  63.48647393313754}, {30.62554452606222,  60.101485548798514, 37.063824618295754, 71.04525498101337}, {56.032005794131614, 71.80616208209968,  67.22546752158931,  83.70215276205255}, {20.14317265947747,  73.77798886182363,  34.25432987619779,  87.24104072094495}, {10.507678860183212, 66.06446404977234,  22.91945017863563,  73.50576752587352}, {26.0796380640738,   39.08543029877627,  37.497243272316375, 42.198598580655705}, {58.204665266130036, 58.20119021138755,  66.86094220293387,  61.613651791527374}, {40.43959914994069,  2.5737454435527933, 47.14440867190218,  10.136829689608973}, {81.61166337839565,  57.04686555019882,  82.13024015743876,  60.52557802686094}, {1.1438702774984308, 64.4390551345789,   1.207827079116793,  74.94606495692364}, {22.698477311365394, 31.694032934311718, 23.012351437738243, 34.826851291697004}, {58.23302290469934,  63.09245797513119,  63.89603555830784,  71.13299682623365}, {1.1209075169457285, 81.28342384198416,  2.010664217814431,  85.39246047317187}, {12.031894943077951, 47.03188640891187,  17.157531829906453, 58.84050109551066}, {25.175447117884868, 53.84501614745653,  29.018643250506607, 59.38873449198591}, {2.2848309030370015, 13.908167333298184, 9.169561431787841,  19.16049137202979}, {50.013550661499245, 78.5109200392331,   61.27884750099618,  90.82242857844415}, {60.35181123067779,  50.30720879159393,  66.40423614499642,  62.711248070454005}, {12.818633233242565, 80.69085735063159,  25.51374909020891,  93.22537975149076}, {13.89435574446365,  30.374627423660982, 26.014177608552792, 40.22893652344269}, {68.59949104329682,  71.57717815724429,  71.14413101711249,  81.32143731631942}, {8.759053910523154,  40.17136447593845,  22.076247428918848, 51.97034411093291}, {75.0237223114521,   10.812195153356786, 75.45859644475163,  24.680056123348074}, {37.640987086884465, 44.31736944555115,  46.79079124130418,  52.298119297002756}, {77.86465045295246,  69.74685405122065,  91.0727578759392,   81.32602647164121}, {41.571023531510896, 41.188931957868,    47.81613155473583,  53.78551712929363}, {46.21623238891625,  12.566288400974617, 60.42998852835609,  23.520076065312416}, {39.651498265328506, 13.503482197678323, 50.2456922936693,   17.970333385957133}, {22.002987425318885, 4.223514231931571,  24.39665459195155,  17.79996696134728}, {10.238509846935935, 17.775671898372956, 24.90139389081459,  30.900047607940877}, {11.945673076143192, 11.005643838128806, 14.458677679728162, 25.935774067123525}, {34.15254570484473,  32.9087837466544,   39.806374568647804, 45.792474254223166}, {1.2619249479259986, 73.38259039620652,  5.732709854315865,  82.08100065666045}, {68.88687814624431,  70.06499982957165,  70.86758866753506,  78.39070584782843}, {53.346140703038856, 38.61621943306142,  58.18001677406793,  46.227279405415416}, {60.91283806646173,  5.328797186659199,  70.97382774644399,  11.165367727083606},
        };
        //@formatter:on
        //queries
        //nothing
        auto query1 = mbr::MBR<double>{81.59858271428983, 88.95212575682031, 87.00714129337072, 92.42905627194374};
        auto query2 = mbr::MBR<double>{82.17807113347706, 83.15724156494792, 87.39346690616222, 84.70254401611389};
        auto query3 = mbr::MBR<double>{84.10969919743454, 72.14696160039038, 86.23449006778775, 79.10082263063724};
        auto query4 = mbr::MBR<double>{21.298871774427138, 1.1709155631470283, 36.23985259304277, 20.747325333798532};
        auto query5 = mbr::MBR<double>{0., 0., 100, 100};
        auto query6 = mbr::MBR<double>{182.17619056720642, 15.748541593521262, 205.43811579298725, 65.97783146157896};

        auto tree = NewRTree<double>(9);
        auto bulk_tree = NewRTree<double>(9);

        std::vector<rtree::Item<double>> bulk_items;
        for (size_t i = 0; i < data.size(); i++) {
            bulk_items.emplace_back(rtree::Item{i + 222222, data[i]});
        }
        bulk_tree.load(bulk_items);

        std::vector<rtree::Item<double>> data_oneByone{};
        for (size_t i = 0; i < data.size(); i++) {
            data_oneByone.emplace_back(rtree::Item{i, data[i]});
        }
        for (auto item: data_oneByone) {
            tree.insert(item);
        }

        auto res1 = tree.search(query1);
        auto res2 = tree.search(query2);
        auto res3 = tree.search(query3);
        auto res4 = tree.search(query4);
        auto res5 = tree.search(query5);
        auto res6 = tree.search(query6);

        REQUIRE(len(res1) == 0);
        REQUIRE(len(res2) == 0);
        REQUIRE(len(res3) == 2);
        REQUIRE(len(res4) == 6);
        REQUIRE(len(res5) == len(data));
        REQUIRE(len(res6) == 0);
        REQUIRE(len(tree.all()) == len(data));
    }
}
