#include "util.h"

#ifndef RTREE_ITEM_H
#define RTREE_ITEM_H
namespace rtree {
    template<typename U>
    struct Item {
        Id id{null_id};
        mbr::MBR<U> box = empty_mbr<U>();

        Item() = default;

        Item(Id _id, mbr::MBR<U> _box) : id(_id), box(std::move(_box)) {
        }

        [[using gnu : const, always_inline, hot]]
        bool operator<(const Item<U> &other) {
            return box < other.box;
        }

        [[using gnu : const, always_inline, hot]] [[nodiscard]]
        mbr::MBR<U> &bbox() { return box; }
    };
}
#endif //RTREE_ITEM_H
