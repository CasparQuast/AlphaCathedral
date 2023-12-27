// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_CATHEDRAL_H_
#define OPEN_SPIEL_GAMES_CATHEDRAL_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <queue>
#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel
{
    namespace cathedral
    {

        inline constexpr int num_players = 2;
        inline constexpr int board_width = 10;
        inline constexpr int board_height = 10;
        inline constexpr int board_size = board_width * board_height;
        inline constexpr int range_of_nn_distribution = 5600; // highest possible encoded move
        inline constexpr int max_game_length = 40; // 30 pieces in total + 10 for possible replaced pieces
        inline constexpr int max_rotations = 4;
        inline constexpr int total_planes = 14 /*pieces*/ + 1 /*empty*/+ 1 /*game_phase represented by normalized move count*/;

        enum class BuildingType
        {
            Tavern,
            Stable,
            Inn,
            Bridge,
            Manor,
            Square,
            Black_Abbey,
            White_Abbey,
            Black_Academy,
            White_Academy,
            Infirmary,
            Castle,
            Tower,
            Cathedral
        };

        const std::array<BuildingType,9> common_buildings = {
            BuildingType::Tavern,
            BuildingType::Stable,
            BuildingType::Inn,
            BuildingType::Bridge,
            BuildingType::Manor,
            BuildingType::Square,
            BuildingType::Infirmary,
            BuildingType::Castle,
            BuildingType::Tower
        };

        const std::vector<BuildingType> black_specific_buildings = {
            BuildingType::Black_Abbey, BuildingType::Black_Academy
        };

        const std::vector<BuildingType> white_specific_buildings = {
            BuildingType::Cathedral, BuildingType::White_Abbey, BuildingType::White_Academy
        };

        enum class Rotation
        {
            ROTATE_0,
            ROTATE_90,
            ROTATE_180,
            ROTATE_270
        };

        std::ostream& operator<<(std::ostream& os, const Rotation& rotation);

        enum class Turnable {
            NO = 0,      // 0
            HALF = 1,    // (0, 90)
            FULL = 3     // (0, 90, 180, 270)
        };

        enum class CellState {
            EMPTY,
            BLUE,
            BLACK,
            BLACK_REGION,
            WHITE,
            WHITE_REGION
        };

        // Factors for rotation calculation for 0, 90, 180, 270 degrees
        constexpr int dx[4] = {1, 0, -1, 0};  
        constexpr int dy[4] = {0, 1, 0, -1};

        struct Square
        {
            int x, y;

            Square(int x, int y) : x(x), y(y) {}

            Square rotate(Rotation rotation) const;

            Square operator+(const Square& other) const;
            bool operator==(const Square& other) const;
            bool operator<(const Square& other) const;
        };

        class Building {

        public:

            Building(int how_many, Turnable turnable, const std::vector<Square>& form)
            : how_many_(how_many), turnable_(turnable), default_form_(form) {
                pre_calculate_forms();
                pre_calculate_corners();
            }

            static const Building& get_instance(BuildingType type);

            const std::vector<Square> form(Rotation rotation, Square pos) const;
            const std::vector<Square> corners(Rotation rotation, Square pos) const;

            inline const int how_many() const { return how_many_; }
            inline const Turnable turnable() const { return turnable_; }

            std::vector<Square> default_form_;

        private:

            static std::vector<Building> create_instances();

            int how_many_;
            Turnable turnable_;

            std::vector<std::vector<Square>> pre_calculated_forms_;
            std::vector<std::vector<Square>> pre_calculated_corners_;

            std::vector<Square> rotate_form(const std::vector<Square> &form, Rotation rotation) const;

            const std::vector<Square>& form(Rotation rotation) const;
            const std::vector<Square>& corners(Rotation rotation) const;

            const std::vector<Square> translate_positions(const std::vector<Square>& positions, Square pos) const;

            void pre_calculate_forms();
            void pre_calculate_corners();
        };

        class Move
        {
            public:

            Move(const Square& position, const BuildingType& type, Rotation rotation);
            Move(int action) : Move(decode_move(action)) {}
        
            int encode() const;
            Move decode_move(int action);

            bool operator==(const Move& other) const;

            std::vector<Square> form;
            std::vector<Square> corners;
            Square pos;
            BuildingType building_type;
            Rotation rotation;
        };

        struct PlayerMove {
            Player player;
            Move move;
        };

        const CellState get_square_color(const Move& move, Player player);

        // Functions for communication with java client (parsing moves from cpp game to java and vice versa)
        const int player_building_to_java_building_id(BuildingType type, int player);
        const std::tuple<BuildingType, int> java_building_to_cpp_building_player(int java_building_id);
        const Rotation parse_rotation_angle(int angle);

        class Playerpieces {

        public:

            explicit Playerpieces(const std::vector<BuildingType>& specific_building_types);

            void use_building(BuildingType type);
            void return_building(BuildingType type);
            bool is_building_available(BuildingType type) const;
            const std::vector<BuildingType> get_available_building_types() const;
            void reset_building_availability();

        private:
            std::array<int, 14> available_buildings_{};
            std::vector<BuildingType> specific_building_types_;

            void initialize_building_availability();
        };

        class Board {

        public:

            Board();

            bool is_move_valid(const Move& move, Player player) const;
            void place_color(const std::vector<Square>& form, CellState color);
            bool is_on_board(const Square& square) const;
            bool color_is_compatible(CellState on_position, CellState to_place) const;

            std::string to_string() const;

            std::array<std::array<CellState, 10>, 10> field;
        };

        class CathedralState : public State
        {

        public:
            CathedralState(std::shared_ptr<const Game> game);

            CathedralState(const CathedralState &) = default;
            CathedralState &operator=(const CathedralState &) = default;

            Player CurrentPlayer() const override
            {
                return IsTerminal() ? kTerminalPlayerId : current_player_;
            }

            std::string ActionToString(Player player, Action action_id) const override;
            std::string ToString() const override;
            bool IsTerminal() const override;
            std::vector<double> Returns() const override;
            std::string InformationStateString(Player player) const override;
            std::string ObservationString(Player player) const override;
            void ObservationTensor(Player player,
                                   absl::Span<float> values) const override;
            std::unique_ptr<State> Clone() const override;
            void UndoAction(Player player, Action move) override;
            std::vector<Action> LegalActions() const override;

            //  MINE

            const std::vector<Move> get_possible_moves() const;
            const std::vector<Move> get_possible_moves(Player player_index) const;
            const std::vector<Move> get_possible_moves(const BuildingType &type, Player player) const;

            void undo_move();
            const bool make_move(const Move& move);
            void make_unvalidated_move(const Move& move);
            const Board& get_board() const { return board; }
            const bool is_finished() const;
            void update_current_player();
            const int get_move_count() const { return history_.size(); }

            std::array<Playerpieces, 2> players;

            Board initial_board;

        protected:
            void DoApplyAction(Action move) override;

        private:

            const std::vector<Move> generate_all_possible_moves() const;
            std::tuple<int, int> calc_score() const;
            bool remove_move(const PlayerMove& move);
            bool place_building(const Move& move);
            void process_region(const std::vector<Square>& region, CellState color);
            void build_regions();
            const std::vector<PlayerMove> get_all_enemy_buildings_in_region(const std::vector<Square>& region, CellState enemyColor) const;
            bool is_in_region(const Square& position, const std::vector<Square>& region) const;
            CellState get_sub_color(CellState color) const;
            const bool is_piece_removed(const PlayerAction& move) const;

            void PopulatePiecePlanes(TensorView<3>& view, Player player) const;
            void PopulateGameProgressPlane(TensorView<3>& view) const;
            void PopulateFreeSquaresPlane(TensorView<3>& view, Player player) const;

            std::vector<PlayerMove> removed_moves;

            Board board;

            Player current_player_ = 0;
            Player outcome_ = kInvalidPlayer;
        };

        
        // Game object.
        class CathedralGame : public Game
        {
        public:
            explicit CathedralGame(const GameParameters &params);

            int NumDistinctActions() const override { return range_of_nn_distribution; }

            std::unique_ptr<State> NewInitialState() const override
            {
                return std::unique_ptr<State>(new CathedralState(shared_from_this()));
            }

            int NumPlayers() const override { return num_players; }

            double MinUtility() const override { return -1; }
            absl::optional<double> UtilitySum() const override { return 0; }
            double MaxUtility() const override { return 1; }

            std::vector<int> ObservationTensorShape() const override
            {
                return {total_planes, board_width, board_height};
            }

            int MaxGameLength() const override { return max_game_length; }

            std::string ActionToString(Player player, Action action_id) const override;
        };

    } 
} 

#endif
