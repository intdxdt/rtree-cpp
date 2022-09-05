#include <limits>
#include <memory>
#include <utility>
#include <tuple>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include "item.h"

#ifndef RTREE_CPP_NODE_H
#define RTREE_CPP_NODE_H

//Node
namespace rtree {
    ///Node type for internal node
    template<typename U>
    struct Node {
        Item<U> item;
        size_t height{};
        bool leaf{};
        mbr::MBR<U> bbox = empty_mbr<U>();
        std::vector<Node<U>> children;

        Node() = default;

        Node(Item<U> item, size_t height, bool leaf, mbr::MBR<U> bbox) :
                item(item), height(height), leaf(leaf), bbox(bbox) {
            children = std::vector<Node<U>>{};
        };

        Node(Node<U> &&other) noexcept:
                item(other.item), height(other.height), leaf(other.leaf), bbox(other.bbox) {
            children = std::move(other.children);
        }

        ~Node() = default;

        Node &operator=(Node<U> &&other) noexcept {
            item = other.item;
            leaf = other.leaf;
            height = other.height;
            bbox = other.bbox;
            children = std::move(other.children);
            return *this;
        }

        void add_child(Node<U> &&child) {
            children.emplace_back(std::move(child));
        }

        Item<U> get_item() {
            return item;
        }
    };

    ///KObj instance struct
    template<typename U>
    struct KObj {
        Node<U> *node;
        mbr::MBR<U> bbox;
        bool is_item;
        double dist;

        double score() {
            return this->dist;
        }

        Item<U> get_item() {
            return this->node->get_item();
        }

        std::string string() {
            std::ostringstream ss;
            ss << this->node->bbox.wkt() << " -> " << this->dist;
            return ss.str();
        }
    };

    template<typename U>
    struct kobj_cmp {
        inline bool operator()(const KObj<U> &a, const KObj<U> &b) const {
            return a.dist > b.dist;
        }
    };

    template<typename U>
    struct x_node_path {
        inline bool operator()(Node<U> &a, Node<U> &b) {
            return a.bbox.minx < b.bbox.minx;
        }
    };

    template<typename U>
    struct y_node_path {
        inline bool operator()(Node<U> &a, Node<U> &b) {
            return a.bbox.miny < b.bbox.miny;
        }
    };

    template<typename T>
    struct xy_node_path {
        inline bool operator()(T &a, T &b) {
            auto d = a.bbox().minx - b.bbox().minx;
            if (feq(d, 0)) {
                d = a.bbox().miny - b.bbox().miny;
            }
            return d < 0;
        }
    };

    //compareNodeMinX computes change in minimum x
    template<typename U>
    inline double compare_minx(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        return a.minx - b.minx;
    }

    //compareNodeMinY computes change in minimum y
    template<typename U>
    inline double compare_miny(const mbr::MBR<U> &a, const mbr::MBR<U> &b) {
        return a.miny - b.miny;
    }

    template<typename U>
    Node<U> NewNode(
            Item<U> item, size_t height, bool leaf,
            std::vector<Node<U>> &&children) {
        mbr::MBR<U> box = empty_mbr<U>();
        if (item.id != null_id) {
            box = item.bbox();
        }
        auto node = Node<U>(item, height, leaf, box);
        node.children = std::move(children);
        return Node<U>(std::move(node));
    }

    template<typename U>
    Node<U> NewNode(Item<U> item, size_t height, bool leaf) {
        mbr::MBR<U> box = empty_mbr<U>();
        if (item.id != null_id) {
            box = item.bbox();
        }
        return {item, height, leaf, box};
    }

    //NewNode creates a new node
    template<typename U>
    Node<U> new_leaf_Node(Item<U> item) {
        return NewNode(item, 1, true, std::vector<Node<U>>{});
    }

    //Constructs children of node
    template<typename U>
    std::vector<Node<U>> make_children(std::vector<Item<U>> &items) {
        std::vector<Node<U>> chs;
        auto n = items.size();
        chs.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            chs.emplace_back(new_leaf_Node<U>(items[i]));
        }
        return chs;
    }

    //dist_bbox computes min bounding rectangle of node children from k to p-1.
    template<typename N, typename U>
    mbr::MBR<U> dist_bbox(N *node, size_t k, size_t p) {
        auto bbox = empty_mbr<U>();
        for (auto i = k; i < p; i++) {
            extend(bbox, node->children[i].bbox);
        }
        return bbox;
    }


    template<typename U>
    Node<U> *node_at_index(std::vector<Node<U>> &a, size_t i) {
        if (a.empty() || (i > a.size() - 1)) {
            return nullptr;
        }
        return &a[i];
    }

    template<typename U>
    Node<U> *node_at_index(std::vector<Node<U> *> &a, size_t i) {
        if (a.empty() || (i > a.size() - 1)) {
            return nullptr;
        }
        return a[i];
    }

    //calculate_bbox calculates its bbox from bboxes of its children.
    template<typename N, typename U>
    void calculate_bbox(N *node) {
        node->bbox = dist_bbox<N, U>(node, 0, node->children.size());
    }

    // adjust bboxes along the given tree path
    template<typename U>
    void adjust_parent_bboxes(const mbr::MBR<U> &bbox, std::vector<Node<U> *> &path, size_t &level) {
        auto n = static_cast<size_t>(-1);
        for (auto i = level; i != n; i--) {
            extend(path[i]->bbox, bbox);
        }
    }

    //_chooseSubtree select child of node and updates path to selected node.
    template<typename U>
    Node<U> *
    choose_subtree(const mbr::MBR<U> &bbox, Node<U> *node, size_t &level, std::vector<Node<U> *> &path) {
        Node<U> *child{nullptr};
        Node<U> *targetNode{nullptr};
        double minArea, minEnlargement, area, enlargement;

        while (true) {
            path.emplace_back(node);
            if (node->leaf || (path.size() - 1 == level)) {
                break;
            }
            minEnlargement = std::numeric_limits<double>::infinity();
            minArea = std::numeric_limits<double>::infinity();

            for (auto &o: node->children) {
                child = &o;
                area = bbox_area(child->bbox);
                enlargement = enlarged_area(bbox, child->bbox) - area;

                // choose entry with the least area enlargement
                if (enlargement < minEnlargement) {
                    minEnlargement = enlargement;
                    if (area < minArea) {
                        minArea = area;
                    }
                    targetNode = child;
                }
                else if (feq(enlargement, minEnlargement)) {
                    // otherwise choose one with the smallest area
                    if (area < minArea) {
                        minArea = area;
                        targetNode = child;
                    }
                }
            }

            node = (targetNode == nullptr) ? &(node->children[0]) : targetNode;
        }
        return node;
    }
}


#endif //RTREE_CPP_NODE_H
