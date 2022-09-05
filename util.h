#include <limits>
#include <memory>
#include <utility>
#include <tuple>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>

#ifndef RTREE_CPP_UTIL_H
#define RTREE_CPP_UTIL_H
namespace rtree {
    using SortBy = std::size_t;
    constexpr SortBy by_x = 0;
    constexpr SortBy by_y = 1;
    using Id = std::size_t;
    static const Id null_id = static_cast<Id>(-1);

    template<typename U>
    std::array<U, 4> empty_bounds() {
        if constexpr (std::is_integral<U>::value) {
            return {
                    std::numeric_limits<U>::max(), std::numeric_limits<U>::max(),
                    std::numeric_limits<U>::min(), std::numeric_limits<U>::min()
            };
        }
        else {
            return {
                    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()
            };
        };

    }

    //@formatter:off
    template<typename U>
    inline mbr::MBR<U> empty_mbr() {
        return {empty_bounds<U>(), true};
    }

    template<typename T>
    inline size_t len(const std::vector<T> &v) {
        return v.size();
    }

    template<typename T>
    T min(const T a, const T b) {
        if constexpr (std::is_integral<T>::value) {
            return b < a ? b : a;
        }
        else {
            return std::fmin(a, b);
        };
    }

    template<typename T>
    T max(const T a, const T b) {
        if constexpr (std::is_integral<T>::value) {
            return b > a ? b : a;
        }
        else {
            return std::fmax(a, b);
        };
    }

    template<typename T>
    T pop(std::vector<T> &a) {
        if (a.empty()) {
            return nullptr;
        }
        auto v = a.back();
        a.resize(a.size() - 1);
        return std::move(v);
    }

    ///std::optional<size_t>
    template<>
    size_t pop(std::vector<size_t> &a) {
        auto v = a.back();
        a.resize(a.size() - 1);
        return v;
    }

    template<typename T>
    std::vector<T> slice(const std::vector<T> &v, size_t i = 0, size_t j = 0) {
        std::vector<T> s(v.begin() + i, v.begin() + j);
        return s;
    }

    template<typename T>
    inline void swap_item(std::vector<T> &arr, size_t i, size_t j) {
        std::swap(arr[i], arr[j]);
    }


    ///slice index
    std::optional<size_t> slice_index(size_t limit, const std::function<bool(size_t)> &predicate) {
        bool bln{false};
        size_t index{0};

        for (size_t i = 0; !bln && i < limit; ++i) {
            index = i;
            bln = predicate(i);
        }
        return bln ? std::optional<size_t>{index} : std::nullopt;
    }


    ///extend bounding box
    template<typename U>
    mbr::MBR<U> &extend(mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        a.minx = min(a.minx, b.minx);
        a.miny = min(a.miny, b.miny);
        a.maxx = max(a.maxx, b.maxx);
        a.maxy = max(a.maxy, b.maxy);
        return a;
    }

    ///computes area of bounding box
    template<typename U>
    double bbox_area(const mbr::MBR<U> &a) {
        return (a.maxx - a.minx) * (a.maxy - a.miny);
    }

    ///computes box margin
    template<typename U>
    double bbox_margin(const mbr::MBR<U> &a) {
        return (a.maxx - a.minx) + (a.maxy - a.miny);
    }

    ///computes enlarged area given two mbrs

    template<typename U>
    double enlarged_area(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        return (max(a.maxx, b.maxx) - min(a.minx, b.minx)) *
               (max(a.maxy, b.maxy) - min(a.miny, b.miny));
    }

    ///contains tests whether a contains b
    template<typename U>
    [[using gnu : const, always_inline, hot]]
    inline bool contains(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        return a.contains(b);
    }

    ///intersects tests a intersect b (mbr)
    template<typename U>
    [[using gnu : const, always_inline, hot]]
    inline bool intersects(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        return a.intersects(b);
    }

    ///computes the intersection area of two mbrs
    template<typename U>
    double intersection_area(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        if (a.disjoint(b)) {
            return 0.0;
        }
        auto minx = (b.minx > a.minx) ? b.minx : a.minx;
        auto miny = (b.miny > a.miny) ? b.miny : a.miny;
        auto maxx = (b.maxx < a.maxx) ? b.maxx : a.maxx;
        auto maxy = (b.maxy < a.maxy) ? b.maxy : a.maxy;
        return (maxx - minx) * (maxy - miny);
    }
}
#endif //RTREE_CPP_UTIL_H
