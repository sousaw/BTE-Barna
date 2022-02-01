// Copyright 2015-2020 The ALMA Project Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.


/// @file
///
/// Definitions of geometry_2d.hpp

#include <geometry_2d.hpp>
#include <io_utils.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <utilities.hpp>

namespace alma {

Eigen::MatrixXd geometry_2d::bounding_box(polygon& p) {
    Eigen::Matrix2d limits;
    Eigen::MatrixXd points;
    std::size_t cols = 0;
    boost::geometry::for_each_point(p, [&cols](point& i) { cols++; });
    points.resize(2, cols);
    cols = 0;
    boost::geometry::for_each_point(p, [&cols, &points](point& i) {
        points(0, cols) = boost::geometry::get<0>(i);
        points(1, cols) = boost::geometry::get<1>(i);
        cols++;
    });
    // Xmin
    limits(0, 0) = points.row(0).minCoeff();
    // Xmax
    limits(1, 0) = points.row(0).maxCoeff();
    // Ymin
    limits(0, 1) = points.row(1).minCoeff();
    // Ymax
    limits(1, 1) = points.row(1).maxCoeff();

    return limits;
}

void geometry_2d::get_sides(Eigen::MatrixXd& vertices_) {
    if (vertices_.rows() != 2) {
        throw alma::input_error("Error in provided vertices,"
                                " bad dimension\n");
    }
    for (auto i = 0; i < vertices_.cols(); i++) {
        Eigen::Vector2d p1 = vertices_.col(i);
        for (auto j = i; j < vertices_.cols(); j++) {
            Eigen::Vector2d p2 = vertices_.col(j);
            /// If same point pass
            if ((p1 - p2).norm() < 1.0e-6)
                continue;
            segment s({p1(0), p1(1)}, {p2(0), p2(1)});
            point sc;
            /// Looking for midpoint
            boost::geometry::centroid(s, sc);
            /// To prevent numerical issues about boder
            /// we slightly displace the point
            /// away from hull center
            Eigen::Vector2d ncsc;
            ncsc << sc.get<0>() - this->center(0),
                sc.get<1>() - this->center(1);

            if (ncsc.norm() == 0.) {
                continue;
            }

            double nc = 1. / ncsc.norm();
            /// Normalize and multiply by the eps
            ncsc(0) *= 1.0e-4 * nc;
            ncsc(1) *= 1.0e-4 * nc;
            point fp(ncsc(0) + sc.get<0>(), ncsc(1) + sc.get<1>());

            bool valid_segment = boost::geometry::covered_by(fp, this->hull);
            /// If valid segment
            if (!valid_segment) {
                geom2d_border b;
                b.p1 = p1;
                b.p2 = p2;
                b.nd = (p2 - p1);
                // Build it to point outside
                Eigen::Vector2d np_t;
                np_t << b.nd(1), -b.nd(0);
                if (np_t.dot(ncsc) < 0.) {
                    np_t *= -1;
                }
                b.np = np_t / np_t.norm();

                auto pa = (p1 - 1.0e-6 * b.nd).eval();
                auto pb = (p2 + 1.0e-6 * b.nd).eval();

                segment sf({pa(0), pa(1)}, {pb(0), pb(1)});

                b.sb = s;
                b.sbl = sf;

                if (std::find_if(borders.begin(),
                                 borders.end(),
                                 [&b](const geom2d_border& a) -> bool {
                                     if ((a.np - b.np).norm() < 1.0e-6)
                                         return true;
                                     return false;
                                 }) == borders.end()) {
                    borders.push_back(b);
                }

                if (borders.size() >
                    static_cast<std::size_t>(vertices_.cols())) {
                    throw alma::geometry_error("To much borders");
                }
            }
        }
    }
}


geometry_2d::geometry_2d(Eigen::MatrixXd& vertices_) {
    //     if (vertices_.cols()!=3) {
    //         std::cerr << "Error in vertices, we are only accepting
    //         triangles\n"; exit(EXIT_FAILURE);
    //     }


    polygon poly;

    for (auto i = 0; i < vertices_.cols(); i++) {
        double x = vertices_(0, i);
        double y = vertices_(1, i);
        boost::geometry::append(poly, boost::make_tuple(x, y));
    }
    /// Creating convex hull
    boost::geometry::convex_hull(poly, this->hull);
    /// Calculating centroid
    point c;
    boost::geometry::centroid(this->hull, c);
    this->center << c.get<0>(), c.get<1>();
    /// Calculating area
    this->area = boost::geometry::area(this->hull);
    /// Geting borders
    get_sides(vertices_);
    /// Getting bounding_box
    this->bbox = bounding_box(this->hull);
}

bool geometry_2d::inside(Eigen::Vector2d& point2check) const {
    point p(point2check(0), point2check(1));
    return boost::geometry::covered_by(p, this->hull);
}

std::tuple<double, Eigen::Vector2d, std::vector<int>>
geometry_2d::get_inter_side(Eigen::Vector2d& r0,
                            Eigen::Vector2d& v,
                            double dt) {
    point p0(r0(0), r0(1));
    point pf(r0(0) + dt * v(0), r0(1) + dt * v(1));
    segment s(p0, pf);

    double time = 1.0e+7;
    Eigen::Vector2d rf;
    int border_id = 0;
    std::vector<int> bids;
    bool already_there = true;

    for (auto& b : borders) {
        /// To rule out already there stuff
        if (boost::geometry::intersects(s, b.sbl) and
            !boost::geometry::covered_by(p0, b.sbl)) {
            if (b.np.dot(v) > 0.) {
                std::vector<point> rinter;
                boost::geometry::intersection(s, b.sbl, rinter);

                double newtime =
                    std::max({(std::abs(v(0)) < 1.0e-6)
                                  ? -1
                                  : (rinter[0].get<0>() - r0(0)) / v(0),
                              (std::abs(v(1)) < 1.0e-6)
                                  ? -1
                                  : (rinter[0].get<1>() - r0(1)) / v(1)});

                /// Corner case
                if (newtime == time) {
                    bids.push_back(border_id);
                }


                if (newtime < time) {
                    bids.clear();
                    rf(0) = rinter[0].get<0>();
                    rf(1) = rinter[0].get<1>();
                    time = newtime;
                    already_there = false;
                    bids.push_back(border_id);
                }
            }
        }

        border_id++;
    }

    /// In the case already there and going outside
    /// of box
    if (already_there) {
        point p0p(r0(0) - 1.0e-6 * v(0), r0(1) - 1.0e-6 * v(1));
        segment sp(p0p, pf);

        border_id = 0;

        for (auto& b : borders) {
            /// To rule out already there stuff
            if (boost::geometry::intersects(sp, b.sbl)) {
                if (b.np.dot(v) > 0.) {
                    std::vector<point> rinter;
                    boost::geometry::intersection(sp, b.sbl, rinter);

                    double newtime =
                        std::max({(std::abs(v(0)) < 1.0e-6)
                                      ? -1.0e+8
                                      : (rinter[0].get<0>() - r0(0)) / v(0),
                                  (std::abs(v(1)) < 1.0e-6)
                                      ? -1.0e+8
                                      : (rinter[0].get<1>() - r0(1)) / v(1)});

                    /// Corner case
                    if (newtime == time) {
                        bids.push_back(border_id);
                    }


                    if (std::abs(newtime) < time) {
                        bids.clear();
                        rf(0) = rinter[0].get<0>();
                        rf(1) = rinter[0].get<1>();
                        time = newtime;
                        bids.push_back(border_id);
                    }
                }
            }

            border_id++;
        }

        if (time > 1.0e+6) {
            //             std::cout << "Error in get_inter_side\n";
            //             std::cout << time << '\t' << dt << std::endl;
            //             std::cout << "v\n" << v << std::endl;
            //             std::cout << "r0\n" << r0 << std::endl;
            //             for (auto &b : borders) {
            //                 std::cout << "nb\n"<< b.np << std::endl;
            //                 std::cout << std::boolalpha <<
            //                 boost::geometry::intersects(sp, b.sbl) << '\t'
            //                 << boost::geometry::intersects(s, b.sbl) << '\t'
            //                 << boost::geometry::covered_by(pf,b.sb) << '\t'
            //                 << boost::geometry::covered_by(p0,b.sb) <<
            //                 std::endl;
            //             }
            //             std::cout << "box_id: " << this->get_id() <<
            //             std::endl;
            // 	    Eigen::Vector2d rf_ = r0+dt*v;
            //             std::cout << std::boolalpha <<
            //                 this->inside(r0) << '\t'  << std::endl;
            //             std::cout << "rf:\n" << rf_ << std::endl <<
            //             std::boolalpha <<
            //                 this->inside(rf_) << '\t' << std::endl;
            throw alma::geometry_error("Error to huge time");
        }
    }

    /// This is to clean out some numerical noise
    /// from geometric library
    if (alma::almost_equal(time, 0.))
        time = 0.;

    if (time < 0.) {
        std::cout << "Error in get_inter_side\n";
        std::cout << time << std::endl;
        std::cout << "v\n" << v << std::endl;
        std::cout << "r0\n" << r0 << std::endl;
        std::cout << "rf\n" << rf << std::endl;
        throw alma::geometry_error("Bad time");
    }

    return std::make_tuple(time, rf, bids);
}

void geometry_2d::calculate_contacts(std::vector<geometry_2d>& system) {
    /// First get real contacts
    for (auto& element : system) {
        if (element.id == this->id)
            continue;
        if (boost::geometry::intersects(this->hull, element.hull)) {
            std::vector<point> pinter;
            /// This gives the 2 points of the segment
            boost::geometry::intersection(this->hull, element.hull, pinter);

            /// Border object
            geom2d_border cb;

            /// If point contact
            if (pinter.size() == 1) {
                cb.p1(0) = pinter[0].get<0>();
                cb.p1(1) = pinter[0].get<1>();
                cb.p2(0) = pinter[0].get<0>();
                cb.p2(1) = pinter[0].get<1>();

                cb.nd = Eigen::Vector2d::Zero();
                cb.sb = segment(pinter[0], pinter[0]);
                cb.np = cb.p1 - this->get_center();
                cb.np /= cb.np.norm();

                contacts.emplace(
                    std::make_pair(element.id, std::vector<geom2d_border>{cb}));
                continue;
            }

            cb.p1(0) = pinter[0].get<0>();
            cb.p1(1) = pinter[0].get<1>();
            cb.p2(0) = pinter[1].get<0>();
            cb.p2(1) = pinter[1].get<1>();

            if ((cb.p2 - cb.p1).norm() < 1.0e-6) {
                cb.p1(0) = pinter[0].get<0>();
                cb.p1(1) = pinter[0].get<1>();
                cb.p2(0) = pinter[1].get<0>();
                cb.p2(1) = pinter[1].get<1>();

                cb.nd = Eigen::Vector2d::Zero();
                cb.sb = segment(pinter[0], pinter[1]);
                cb.np = cb.p1 - this->get_center();
                cb.np /= cb.np.norm();

                contacts.emplace(
                    std::make_pair(element.id, std::vector<geom2d_border>{}));
                continue;
            }

            /// Get director vector of segment
            cb.nd = (cb.p2 - cb.p1); //
                                     //(cb.p2 - cb.p1).norm();
            /// Get perpendicular vector pointing outside
            segment s(pinter[0], pinter[1]);
            point sc;
            /// Looking for midpoint
            boost::geometry::centroid(s, sc);
            /// To prevent numerical issues about boder
            /// we slightly displace the point
            /// away from hull center
            Eigen::Vector2d ncsc;
            ncsc << sc.get<0>() - this->center(0),
                sc.get<1>() - this->center(1);
            Eigen::Vector2d np_t;
            np_t << cb.nd(1), -cb.nd(0);
            if (np_t.dot(ncsc) < 0.) {
                np_t *= -1;
            }
            cb.np = np_t / np_t.norm();
            cb.sb = s;

            auto pa = (cb.p1 - 1.0e-6 * cb.nd / cb.nd.norm()).eval();
            auto pb = (cb.p2 + 1.0e-6 * cb.nd / cb.nd.norm()).eval();
            segment sf({pa(0), pa(1)}, {pb(0), pb(1)});
            cb.sbl = sf;

            contacts.emplace(
                std::make_pair(element.id, std::vector<geom2d_border>{cb}));
        }
    }

    if (contacts.size() == 0 and system.size() > 1) {
        throw alma::geometry_error("There are unconnected boxes");
    }
}


/// Helper function definitions
///@param[in] r point to check if tri-or-higher
///         intersection
///@param[in] gs vector containing the geometry
std::pair<bool, std::vector<std::size_t>> in_corner3(
    Eigen::Vector2d& r,
    std::vector<geometry_2d>& gs) {
    std::size_t ic = 0;
    std::vector<std::size_t> ids;
    ids.reserve(gs.size());
    boost::tuple<double, double> p(r(0), r(1));
    for (auto& g : gs) {
        if (boost::geometry::covered_by(p, g.get_poly())) {
            ids.push_back(g.get_id());
            ic++;
        }
    }

    if (ic > 2) {
        return std::make_pair(true, ids);
    }
    return std::make_pair(false, ids);
}

std::vector<alma::geometry_2d> read_geometry_XML(std::string xmlfname) {
    // Vector containing geometry
    std::vector<alma::geometry_2d> geometries;


    // Create empty property tree object
    boost::property_tree::ptree tree;

    // Parse XML input file into the tree
    boost::property_tree::read_xml(xmlfname, tree);


    for (const auto& v : tree.get_child("Geometry")) {
        if (v.first == "number_of_boxes") {
            std::size_t ngeometries =
                alma::parseXMLfield<std::size_t>(v, "Ngeom");
            geometries.reserve(ngeometries);
        }
        /// Iterate throught boxes
        if (v.first == "Box") {
            auto box_tree = v.second;
            std::string matname;
            Eigen::MatrixXd vertices;
            double Teq = -1.;
            bool periodic = false;
            std::size_t box2translate;
            Eigen::Vector2d translation;
            bool reservoir = false;
            double theta = 0.;
            double Treal = -1.;

            for (auto it = box_tree.begin(); it != box_tree.end(); it++) {
                /// Parsing name
                if (it->first == "MaterialID")
                    matname = alma::parseXMLfield<std::string>(*it, "name");
                // Parsing vertices:
                if (it->first == "Vertices") {
                    int rows, cols;
                    rows = alma::parseXMLfield<int>(*it, "dim");
                    cols = alma::parseXMLfield<int>(*it, "npoints");
                    vertices.resize(rows, cols);

                    std::stringstream datass(
                        (box_tree).get<std::string>("Vertices"));

                    for (auto i = 0; i < rows * cols; i++) {
                        datass >> vertices.data()[i];
                    }
                }
                /// Equilibrium temperature
                if (it->first == "initCnd") {
                    Teq = alma::parseXMLfield<double>(*it, "Teq");
                    if (alma::probeXMLfield<double>(*it, "Tinit")) {
                        Treal = alma::parseXMLfield<double>(*it, "Tinit");
                    }
                    else {
                        Treal = Teq;
                    }
                }
                /// Parsing periodic:
                if (it->first == "Translate_to") {
                    box2translate = alma::parseXMLfield<std::size_t>(*it, "id");

                    periodic = true;

                    std::stringstream datass(
                        (box_tree).get<std::string>("Translate_to"));

                    datass >> translation(0) >> translation(1);
                }
                /// Reservoir
                if (it->first == "Reservoir")
                    reservoir = true;
                /// Parsing angle:
                if (it->first == "theta")
                    theta = alma::parseXMLfield<double>(*it, "angle");
            }

            /// Set properties
            alma::geometry_2d g(vertices);
            if (periodic) {
                g.periodic = true;
                g.translation = translation;
                g.box2translate = box2translate;
            }
            if (reservoir)
                g.reservoir = true;

            g.Teq = Teq;
            g.Treal = Treal;
            g.material = matname;
            g.theta = theta;
            g.rotmat.setZero();
            /// Obtain rotation matrix
            if (theta == 0.) {
                g.rotmat = Eigen::Matrix2d::Identity();
            }
            else {
                g.rotmat(0, 0) = std::cos(theta);
                g.rotmat(1, 1) = std::cos(theta);
                g.rotmat(0, 1) = -std::sin(theta);
                g.rotmat(1, 0) = std::sin(theta);
            }
            geometries.push_back(g);
        }
    }

    /// Assing ids
    assign_geom_ids(geometries);
    /// Calculate contacts
    for (auto& g : geometries)
        g.calculate_contacts(geometries);

    return geometries;
}

Eigen::MatrixXd calculate_gradientT(std::vector<alma::geometry_2d>& sys) {
    Eigen::MatrixXd gradients(3, sys.size());
    gradients.setZero();


    for (auto& s : sys) {
        if (s.reservoir or s.periodic)
            continue;
        /// Get data for gradient
        /// calculation
        double T0 = s.Teq;
        auto id0 = s.get_id();
        Eigen::Vector2d c0 = s.get_center();

        auto& contacts = s.get_contacts();

        std::size_t csize = 0;
        for (auto& [c, b] : contacts) {
            if (!sys[c].reservoir and !sys[c].periodic)
                csize++;
        }


        /// We want to solve:
        /// f(r0+hi) - f(r0) = h_i * gradf
        /// dfi = hi*gradf
        /// gradf = MPinv(hi)*dfi
        /// where MPinv is the Moore–Penrose inverse
        Eigen::VectorXd dfi(csize);
        Eigen::MatrixXd hi(csize, 2);

        int ci = 0;
        for (auto& [c, b] : contacts) {
            if (sys[c].reservoir or sys[c].periodic)
                continue;
            dfi(ci) = sys[c].Teq - T0;
            Eigen::Vector2d hi_ = sys[c].get_center() - c0;
            hi.row(ci) = hi_;
            ci++;
        }

        /// The actual MPinv is not used
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cqr(hi);

        /// But we solve the minimum-norm solution gradT
        /// to a least squares problem
        Eigen::Vector2d gradT = cqr.solve(dfi);

        gradients(0, id0) = gradT(0);
        gradients(1, id0) = gradT(1);
    }
    return gradients;
}


}; // namespace alma
