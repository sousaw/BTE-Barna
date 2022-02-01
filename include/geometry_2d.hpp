// Copyright 2015-2022 The ALMA Project Developers
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


#pragma once

/// @file
///
/// Code used to manage 2d boxes

#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <random>
#include <utilities.hpp>
#include <pcg_random.hpp>
#include <exceptions.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <mutex>
#include <limits>
#include <map>

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian);

namespace alma {

/// Those are helper functions that turn
/// 3d points to 2d points
///@param xyz - 3d vector to become 2d vector
///@return 2d vector discarding the z axis
inline Eigen::Vector2d xy_me(Eigen::Vector3d& xyz) {
    return xyz.block(0, 0, 2, 1);
}

/// Those are helper functions that turn
/// 3d points to 2d points
///@param xyz - matrix of 3d points to become 2d vector
///@return 2d vectors discarding the z axis
inline Eigen::MatrixXd xy_me(Eigen::MatrixXd& xyz) {
    return xyz.block(0, 0, 2, xyz.cols());
}
/// It appends a 0 to z axis to obtain 3d vector
///@param xy - 2d vector to become 3d vector
///@return 2d vector adding
inline Eigen::Vector3d upDim(Eigen::Vector2d& xy) {
    Eigen::Vector3d xyz;
    xyz << xy(0), xy(1), 0.;
    return xyz;
}


/// This contains info about sides
struct geom2d_border {
    /// Point limits of border
    Eigen::Vector2d p1;
    Eigen::Vector2d p2;
    // The vector defined
    /// as p2-p1
    Eigen::Vector2d nd;
    /// The normalized perpendicular vector
    /// pointing outside
    Eigen::Vector2d np;
    /// The segment object
    boost::geometry::model::segment<boost::tuple<double, double>> sb;
    /// The segment object little bit larger
    boost::geometry::model::segment<boost::tuple<double, double>> sbl;
    /// Properties of border:
    double Tboxeq = -1;
    int Eborder = 0;

    std::mutex door;

    geom2d_border() = default;

    geom2d_border(const geom2d_border& A)
        : p1(A.p1), p2(A.p2), nd(A.nd), np(A.np), sb(A.sb), sbl(A.sbl),
          Tboxeq(A.Tboxeq), Eborder(A.Eborder), door() {
        /// The mutex is not copyied but generated
        /// as new in the copy
    }

    geom2d_border& operator=(const geom2d_border& A) {
        p1 = A.p1;
        p2 = A.p2;
        nd = A.nd;
        np = A.np;
        sb = A.sb;
        sbl = A.sbl;
        Tboxeq = A.Tboxeq;
        Eborder = A.Eborder;
        return *this;
    }


    /// Return a random point in the border
    ///@param[in] r - random generator
    ///@return    3d vector with random position
    template <class Random> Eigen::Vector2d get_random_point(Random& r) {
        Eigen::Vector2d rp =
            this->p1 + std::uniform_real_distribution(0., 1.)(r) * this->nd;
        while (true) {
            boost::tuple<double, double> rp_(rp(0), rp(1));
            if (boost::geometry::covered_by(rp_, sb))
                break;
            rp =
                this->p1 + std::uniform_real_distribution(0., 1.)(r) * this->nd;
        }
        return rp;
    }

    /// Returns the border length in length units
    ///@return border length in length units
    double get_length() const {
        return (p2 - p1).norm();
    }   
};


class geometry_2d {
private:
    /// Some typedefs to make shorter definitions
    typedef boost::tuple<double, double> point;
    typedef boost::geometry::model::polygon<point> polygon;
    typedef boost::geometry::model::segment<point> segment;

    /// Boost convex_hull
    polygon hull;

    /// Central point
    Eigen::Vector2d center;

    /// Area
    double area;
    std::vector<geom2d_border> borders;

    /// Contact map
    /// [other geoms ids,border defining contact region]
    std::map<std::size_t, std::vector<geom2d_border>> contacts;

    /// Geom id
    std::size_t id;

    /// Matrix containing bounding box
    Eigen::MatrixXd bbox;


    /// It returns the bounding box limits
    ///@param[in] p - polygon to search the bounding box from
    ///@returns   the bounding box stored as follows:
    ///           (xmin,ymin)
    ///           (xmax,ymax)
    Eigen::MatrixXd bounding_box(polygon& p);

    /// Get the sides of the figures
    ///@param[in] vertices_ :  Vector containing the vertices
    void get_sides(Eigen::MatrixXd& vertices_);

public:
    /// Public variables about info
    /// Equilibrium temperature
    double Teq = -1.;
    double Treal = -1.;
    /// Is reservoir
    bool reservoir = false;
    /// Material ID
    std::string material;
    /// If periodic
    bool periodic = false;
    /// Translation to apply
    Eigen::Vector2d translation;
    /// Box id to which translate
    std::size_t box2translate;
    /// Angle
    double theta = 0.;
    Eigen::Matrix2d rotmat;

    /// Notice that move semantics is
    /// specifically forbiden
    /// as move Constructor and
    /// move assignment operator
    /// are declared deleted

    /// Default and deleted constructors:
    geometry_2d() = default;
    geometry_2d(const geometry_2d& original) = default;
    geometry_2d(geometry_2d&& source) = delete;
    ~geometry_2d() = default;

    /// Default assignment operators:
    geometry_2d& operator=(const geometry_2d& original) = default;
    geometry_2d& operator=(geometry_2d&& source) = delete;

    /// Constructor
    ///@param[in] vertices_ - Matrix containing 2d points
    ///                       of the triangle vertex
    geometry_2d(Eigen::MatrixXd& vertices_);

    /// Other constructors are left to compiler

    /// It returns true if inside or in border of hull
    ///@param[in] point2check - 2d point to check if inside geometry
    ///@return true if point is inside geometry
    bool inside(Eigen::Vector2d& point2check) const;


    /// It returns the area
    ///@return area in (length unit)**2
    double get_area() const {
        return this->area;
    }

    /// It returns the center
    ///@return center (calculated using mean)
    Eigen::Vector2d get_center() const {
        return this->center;
    }

    /// It returns intersection with polygon sides:
    ///@param[in] r0 - original position
    ///@param[in] v  - velocity vector
    ///@param[in] dt - time step
    ///@return    [time to side collision,
    ///            collision position    ,
    ///            border ids]
    std::tuple<double, Eigen::Vector2d, std::vector<int>>
    get_inter_side(Eigen::Vector2d& r0, Eigen::Vector2d& v, double dt);

    /// Getter for geometry ID
    ///@return geometry id
    std::size_t get_id() {
        return this->id;
    }
    /// Setter for geometry ID
    ///@param[in] vale to set geometry id to
    void set_id(std::size_t id_) {
        this->id = id_;
    }


    /// Builds up contacts from other boxes
    /// Id need to be set up
    ///@param[in] system - geometry_2d of all system
    void calculate_contacts(std::vector<geometry_2d>& system);

    /// Returns borders in general
    ///@return all polygon sides
    std::vector<geom2d_border>& get_borders() {
        return this->borders;
    }

    /// Returns border
    ///@param[in] border_id - the border id
    ///@return specific side of the polygon
    geom2d_border& get_border(std::size_t border_id) {
        return this->borders[border_id];
    }

    /// Modify border Tboxeq variable
    ///@param[in] border_id - the border id
    ///@param[in] _Tboxeq - value to which variable is set
    void set_border_Tboxeq(std::size_t border_id, double _Tboxeq) {
        this->borders[border_id].Tboxeq = _Tboxeq;
    }


    /// Get border Tboxeq variable
    ///@param[in] border_id - the border id
    ///@return id-th side temperature
    double get_border_Tboxeq(std::size_t border_id) {
        return this->borders[border_id].Tboxeq;
    }


    /// Modify border Eborder variable
    ///@param[in] border_id - the border id
    ///@param[in] _Eborder - value to which variable is set
    void set_border_Eborder(std::size_t border_id, double _Eborder) {
        this->borders[border_id].door.lock();
        this->borders[border_id].Eborder = _Eborder;
        this->borders[border_id].door.unlock();
    }

    /// Modify border Eborder variable
    ///@param[in] border_id - the border id
    ///@param[in] _Eborder  - value to add to variable
    void add_border_Eborder(std::size_t border_id, double _Eborder) {
        this->borders[border_id].door.lock();
        this->borders[border_id].Eborder += _Eborder;
        this->borders[border_id].door.unlock();
    }


    /// Get border Eborder variable
    ///@param[in] border_id - the border id
    ///@return id-th boundary energy
    double get_border_Eborder(std::size_t border_id) {
        return this->borders[border_id].Eborder;
    }


    /// Returns the polygon
    ///@return polygon
    polygon& get_poly() {
        return this->hull;
    }

    /// Returns the contacts of the geometry
    ///@return [contact box id,vector(contact region)]
    std::map<std::size_t, std::vector<geom2d_border>>& get_contacts() {
        return this->contacts;
    }

    /// Returns bounding box limits
    ///@return bounding box limits ((xmin,ymin),(xmax,ymax))
    Eigen::MatrixXd get_bbox() {
        return this->bbox;
    }

    /// Get point random point inside the geometry
    ///@param[in] r - random number generator
    ///@return    random coordinates inside geometry
    template <class Random> Eigen::Vector2d get_random_point(Random& r) {
        /// Randomly generate points inside bounding box
        /// then check if inside geometry
        /// Inefficient when the ratio between bbox and
        /// geometry is large
        Eigen::Vector2d rtrial;
        while (true) {
            rtrial(0) =
                this->bbox(0, 0) + std::uniform_real_distribution(0., 1.)(r) *
                                       (this->bbox(1, 0) - this->bbox(0, 0));
            rtrial(1) =
                this->bbox(0, 1) + std::uniform_real_distribution(0., 1.)(r) *
                                       (this->bbox(1, 1) - this->bbox(0, 1));
            point rt(rtrial(0), rtrial(1));
            if (boost::geometry::covered_by(rt, this->get_poly()))
                break;
        }
        return rtrial;
    }


    /// Function for translation:
    ///@param[in] r0   - initial point to translate
    ///@param[in] v    - point direction
    ///@param[in] gs   - vector of geometries
    ///@param[in] rng  - random number generator
    ///@return [new box id,new position after translation]
    template <class Random>
    std::pair<std::size_t, Eigen::Vector2d> translate(
        Eigen::Vector2d& r0,
        Eigen::Vector2d& v,
        std::vector<geometry_2d>& gs,
        Random& rng) {
        /// Check if using it in periodic structure
        if (!gs[this->id].periodic) {
            throw alma::geometry_error("Error: this can only be used in"
                                       "periodic cells");
        }

        /// Get translation vector
        /// and box id to translate
        Eigen::Vector2d T = 1.0005 * gs[this->id].translation;
        auto Tibox = gs[this->id].box2translate;

        if (Tibox >= gs.size() or alma::almost_equal(T.norm(), 0.)) {
            std::cout << this->id << '\t' << Tibox << '\t' << T << std::endl;
            throw alma::geometry_error("Error: Bad id for mapping of"
                                       "periodic cells");
        }

        point p0(r0(0), r0(1));
        point p1(r0(0) + T(0), r0(1) + T(1));
        // Segment:
        segment s(p0, p1);


        geom2d_border sborder;
        for (auto& b : gs[Tibox].get_borders()) {
            if (boost::geometry::intersects(s, b.sbl)) {
                sborder = b;
                break;
            }
        }


        std::vector<point> rinter;
        boost::geometry::intersection(s, sborder.sbl, rinter);

        if (rinter.empty()) {
            std::cout << "r0:\n" << r0 << std::endl;
            std::cout << "v:\n" << v << std::endl;
            std::cout << "T:\n" << T << std::endl;
            std::cout << "rn:\n" << (r0 + T).eval() << std::endl;
            std::cout << "center of newbox:";
            std::cout << gs[Tibox].get_center() << std::endl;
            throw alma::geometry_error("Error in translation, "
                                       "no box found");
        }
        
        Eigen::Vector2d transpoint;
        Eigen::Vector2d transpoint_T;
        Eigen::Vector2d transpoint_v;
        transpoint(0) = rinter[0].get<0>();
        transpoint(1) = rinter[0].get<1>();
        transpoint_T = transpoint -
		 1.0e-6 * gs[this->id].translation /
		 (gs[this->id].translation).norm();
        transpoint_v = transpoint + 1.0e-6 * v/v.norm();
        /// Getting point
        std::vector<Eigen::Vector2d> rf;
        rinter.clear();

        std::vector<std::size_t> pboxes;
        for (auto& [ibc, border] : gs[Tibox].get_contacts()) {
	    bool here_i = boost::geometry::intersects(s,
                                 border[0].sbl);
	    bool here_T = gs[ibc].inside(transpoint);
            bool here_Tv = gs[ibc].inside(transpoint_v);
            bool here_TT = gs[ibc].inside(transpoint_T);

	    bool here = here_i or here_T or here_Tv or here_TT;
 
            if (ibc == Tibox or !here or
               gs[ibc].periodic or gs[ibc].reservoir)
                continue;
            

	    if (here_i) {
                boost::geometry::intersection(s,
                            border[0].sbl,rinter);
                Eigen::Vector2d rf_;
                rf_(0) = rinter[0].get<0>();
                rf_(1) = rinter[0].get<1>();
                rf.push_back(rf_);
                pboxes.push_back(ibc);
            }
            else if (here_T){
		rf.push_back(transpoint);
                pboxes.push_back(ibc);
	    }
	    else if (here_Tv) {
		rf.push_back(transpoint_v);
                pboxes.push_back(ibc);
	    }
            else {
		rf.push_back(transpoint_T);
                pboxes.push_back(ibc);
            }
        }
        /// Choose randomly if we are in corner
        Eigen::Vector2d RF;

        std::size_t newibox;
        if (pboxes.size() > 1) {
            std::vector<std::size_t> pboxes2;

            for (std::size_t ib = 0; ib < pboxes.size(); ib++) {
                Eigen::Vector2d rcheck = rf[ib] + 1.0e-6 * v;
                if (gs[pboxes[ib]].inside(rcheck))
                    pboxes2.push_back(pboxes[ib]);
            }

            auto pos = alma::choose(pboxes2.begin(), pboxes2.end(), rng) -
                       pboxes2.begin();

            RF = rf[pos];
            newibox = pboxes2[pos];
        }
        else if (pboxes.size() == 1) {
            newibox = pboxes[0];
            RF = rf[0];
        }
        else {
            throw alma::geometry_error("Error in Translation");
        }
        return std::make_pair(newibox,RF);
    }
};


/// Helper function to add id to geometries contained in vector
///@param[in] gs vector containing the geometries
inline void assign_geom_ids(std::vector<geometry_2d>& gs) {
    for (std::size_t i = 0; i < gs.size(); i++) {
        gs[i].set_id(i);
    }
}

/// Check if point is in corner shared by 3 polygons or more
/// it also gives the id of those figures
std::pair<bool, std::vector<std::size_t>> in_corner3(
    Eigen::Vector2d& r,
    std::vector<geometry_2d>& gs);

/// Solving intersection in three shared corner
/// We are not making contact with
/// void through single points
template <class Random>
/// It returns intersection with polygon sides:
///@param[in] rf  - position to correct
///@param[in] v   - velocity vector
///@param[in] sys - system geometry
///@param[in] candidates - candidates to where particle can be put
///@param[in] r   - random generator
///@return    [box id,
///            border ids]
std::pair<std::size_t, std::size_t> correct_corner_problem(
    Eigen::Vector2d& rf,
    Eigen::Vector2d& v,
    std::vector<geometry_2d>& sys,
    std::vector<std::size_t>& candidates,
    Random& r) {
    std::map<std::size_t, std::vector<std::size_t>> bids;

    std::vector<std::size_t> finalist;

    boost::tuple<double, double> rc(rf(0), rf(1));

    for (auto candidate : candidates) {
        bids[candidate] = std::vector<std::size_t>(0);
        std::size_t border_id = 0;
        for (auto& b : sys[candidate].get_borders()) {
            /// To rule out already there stuff
            if (boost::geometry::intersects(rc, b.sbl)) {
                if (b.np.dot(v) > 0.)
                    bids[candidate].push_back(border_id);
            }
            border_id++;
        }

        /// In triangles facing exterior it can only
        /// exist single solution
        if (bids[candidate].size() > 1) {
            bids.erase(candidate);
        }
        else {
            finalist.push_back(candidate);
        }
    }

    std::size_t winner = *(alma::choose(finalist.begin(), finalist.end(), r));

    return std::make_pair(winner, bids[winner][0]);
}

/// Returns geometry_2d vector containing all vectors
///@param[in] xmlfname - xml filename
///@return vector containing all geometric data for each box
std::vector<alma::geometry_2d> read_geometry_XML(std::string xmlfname);

/// Calculates the gradient for the box using the contact boxes
///@param[in] sys - system
///@return thermal gradient
Eigen::MatrixXd calculate_gradientT(std::vector<alma::geometry_2d>& sys);


}; // namespace alma
