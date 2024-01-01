#include "open_spiel/games/cathedral/cathedral.h"


#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "cathedral.h"

namespace open_spiel {
namespace cathedral {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"cathedral",
    /*long_name=*/"Cathedral",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CathedralGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace


void CathedralState::DoApplyAction(Action move) {
  make_move(Move(move));
}

std::vector<Action> CathedralState::LegalActions() const {
  std::vector<Action> possible_actions;
  const std::vector<Move>& possible_moves = get_possible_moves();
  for (const Move& move : possible_moves) {
    possible_actions.push_back(move.encode());
  }

  // Sort the possible actions in ascending order
  std::sort(possible_actions.begin(), possible_actions.end());

  return possible_actions;
}


std::string CathedralState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}

CathedralState::CathedralState(std::shared_ptr<const Game> game) : 
State(game), players{Playerpieces(white_specific_buildings), Playerpieces(black_specific_buildings)}, current_player_(0)
{
    board = Board();
}

bool CathedralState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || is_finished();
}

std::vector<double> CathedralState::Returns() const {

    if (is_finished()) {

        const auto& final_score = calc_score();

        int white_score = std::get<0>(final_score);
        int black_score = std::get<1>(final_score);

        if (black_score > white_score)
            return {1.0, -1.0};
        else if (white_score > black_score)
            return {-1.0, 1.0};
    }

    return {0.0, 0.0};
}

std::string CathedralState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CathedralState::ToString() const {
  return board.to_string();
}

std::string CathedralState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CathedralState::ObservationTensor(Player player, absl::Span<float> values) const {

    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);

    TensorView<3> view(values, {total_planes, board_width, board_width}, true);

    // Planes for all unique pieces
    PopulatePiecePlanes(view, player);

    // Normalized move counter to indicate game phase
    PopulateGameProgressPlane(view);

    // Free squares for a given player
    PopulateFreeSquaresPlane(view, player);

    // Current player
    PopulateCurrentPlayerPlane(view, player);
}

void CathedralState::PopulatePiecePlanes(TensorView<3>& view, Player player) const {
    for (const auto& player_action : history_) {

        if (is_piece_removed(player_action))
            continue;

        const auto& move = Move(player_action.action);

        int place_index = static_cast<int>(move.building_type);
        int rotation = static_cast<int>(move.rotation);

        float value = (player_action.player == player ? 1 : -1) * (1.0f + rotation * 0.25f);

        for (const auto& square : move.form) {
            view[{place_index, square.x, square.y}] = value;
        }
    }
}

void CathedralState::PopulateGameProgressPlane(TensorView<3>& view) const {
    float game_phase = static_cast<float>(history_.size()) / max_game_length;
    for (int y = 0; y < board_width; ++y) {
        for (int x = 0; x < board_width; ++x) {
            view[{14, y, x}] = game_phase;
        }
    }   
}

const bool CathedralState::is_piece_removed(const PlayerAction &player_action) const
{
    const PlayerMove& player_move{player_action.player, player_action.action};

    if (std::find(removed_moves.begin(), removed_moves.end(), player_move) != removed_moves.end()) {
        return true;
    }

    return false;
}

void CathedralState::UndoAction(Player player, Action move) {
  undo_move();
  current_player_ = player;
  outcome_ = kInvalidPlayer;
}

std::unique_ptr<State> CathedralState::Clone() const {
  return std::unique_ptr<State>(new CathedralState(*this));
}

std::string CathedralGame::ActionToString(Player player, Action action) const {

    std::ostringstream os;

    const auto& move = Move(action);

    os << player_building_to_java_building_id(move.building_type, player)
       << " " << move.rotation
       << " " << move.pos.x
       << " " << move.pos.y;

    return os.str();
}

CathedralGame::CathedralGame(const GameParameters& params)
    : Game(kGameType, params) {}


Playerpieces::Playerpieces(const std::vector<BuildingType>& specific_building_types)
    : specific_building_types_(specific_building_types) {
    initialize_building_availability();
}

void Playerpieces::use_building(BuildingType type)
{
    if (!is_building_available(type)) {
        throw std::out_of_range("Building is no longer available");
    }

    available_buildings_[static_cast<int>(type)]--;
}

void Playerpieces::return_building(BuildingType type)
{
    if (type != BuildingType::Cathedral)
   	 available_buildings_[static_cast<int>(type)]++;
}

bool Playerpieces::is_building_available(BuildingType type) const 
{
    return available_buildings_[static_cast<int>(type)] > 0;
}

const std::vector<BuildingType> Playerpieces::get_available_building_types() const
{
    std::vector<BuildingType> available_types;

    for (int i = 0; i < available_buildings_.size(); ++i) {
        if (available_buildings_[i] > 0) {
            available_types.push_back(static_cast<BuildingType>(i));
        }
    }

    return available_types;
}

void Playerpieces::reset_building_availability()
{
    initialize_building_availability();
}

void Playerpieces::initialize_building_availability() {
    std::fill(available_buildings_.begin(), available_buildings_.end(), 0);

    for (const auto& type : common_buildings) {
        available_buildings_[static_cast<int>(type)] += Building::get_instance(type).how_many();
    }

    for (const auto& type : specific_building_types_) {
        available_buildings_[static_cast<int>(type)] += Building::get_instance(type).how_many();
    }
}

Board::Board()
{
    for (auto& row : field) {
        row.fill(CellState::EMPTY);
    }
}

bool Board::is_move_valid(const Move &move, Player player) const
{
    CellState building_color = get_square_color(move, player);

    for (const Square& square : move.form) {

        if (!is_on_board(square) ||
            !color_is_compatible(field[square.y][square.x], building_color)) {
            return false; // The placement is not valid
        }
    }

    return true; // The placement is valid
}

bool Board::is_on_board(const Square &square) const
{
    return square.x >= 0 && square.x < 10 && square.y >= 0 && square.y < 10;
}

bool Board::color_is_compatible(CellState on_position, CellState to_place) const {
    return on_position == CellState::EMPTY ||
           (on_position == CellState::BLACK_REGION && to_place == CellState::BLACK) ||
           (on_position == CellState::WHITE_REGION && to_place == CellState::WHITE);
}

void Board::place_color(const std::vector<Square> &form, CellState color)
{
    for (const auto& square : form) {
        field[square.y][square.x] = color;
    }
}

std::string Board::to_string() const {
    std::ostringstream os;
    for (const auto& row : field) {
        for (const auto& cell : row) {
            os << static_cast<int>(cell) << ' ';
        }
        os << '\n';
    }
    return os.str();
}

// [input, output] -> [ROTATE_0, "0"], [ROTATE_90, "90"], [ROTATE_180, "180"], [ROTATE_270, "270"]
std::ostream &operator<<(std::ostream &os, const Rotation &rotation)
{
    return os << std::to_string(static_cast<int>(rotation) * 90);
}

Square Square::rotate(Rotation rotation) const
{
    int rotation_index = static_cast<int>(rotation);

    return Square(
        x * dx[rotation_index] - y * dy[rotation_index],
        y * dx[rotation_index] + x * dy[rotation_index]
    );
}

Square Square::operator+(const Square &other) const
{
    return Square(x + other.x, y + other.y);
}

bool Square::operator==(const Square &other) const
{
    return x == other.x && y == other.y;
}

bool Square::operator<(const Square &other) const
{
    if (y == other.y) return x < other.x;
        return y < other.y;
}

std::vector<Building> Building::create_instances()
{
    std::vector<Building> instances;
    instances.reserve(14);
    instances.emplace_back(2, Turnable::NO, std::vector<Square>{{0, 0}}); //tavern
    instances.emplace_back(2, Turnable::HALF, std::vector<Square>{{0, 0}, {1, 0}}); //stable
    instances.emplace_back(2, Turnable::FULL, std::vector<Square>{{0, 0}, {1, 0}, {1, 1}}); //inn
    instances.emplace_back(1, Turnable::HALF, std::vector<Square>{{0, 0}, {0, -1}, {0, 1}}); //bridge
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, 0}, {0, 0}, {1, 0}, {0, 1}}); //manor
    instances.emplace_back(1, Turnable::NO, std::vector<Square>{{0, 0}, {0, 1}, {1, 0}, {1, 1}}); //square
    instances.emplace_back(1, Turnable::HALF, std::vector<Square>{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}); // black abbey
    instances.emplace_back(1, Turnable::HALF, std::vector<Square>{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}); //white abbey
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, -1}}); //Black academy
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, -1}, {0, -1}, {0, 0}, {0, 1}, {1, 0}}); //white academy
    instances.emplace_back(1, Turnable::NO, std::vector<Square>{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}}); //infirmary
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}, {1, 1}}); //castle
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, -1}, {0, -1}, {0, 0}, {1, 0}, {1, 1}}); //tower
    instances.emplace_back(1, Turnable::FULL, std::vector<Square>{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {1, 0}}); //cathedral

    return instances;
}

const Building& Building::get_instance(BuildingType type) {
    static const std::vector<Building> instances = create_instances();
    return instances[static_cast<int>(type)];
}
const std::vector<Square> Building::form(Rotation rotation, Square pos) const
{
    const auto& rotated_form = form(rotation);
    return translate_positions(rotated_form, pos);
}

const std::vector<Square> Building::corners(Rotation rotation, Square pos) const {
    const auto& rotated_corners = corners(rotation);
    return translate_positions(rotated_corners, pos);
}

std::vector<Square> Building::rotate_form(const std::vector<Square> &form, Rotation rotation) const
{
    std::vector<Square> rotated_form;

    for (const auto& square : form) {
        Square rotated_square = square.rotate(rotation);
        rotated_form.push_back(rotated_square);
    }

    return rotated_form;
}

const std::vector<Square> &Building::form(Rotation rotation) const
{
    return pre_calculated_forms_[static_cast<int>(rotation)];
}

const std::vector<Square> &Building::corners(Rotation rotation) const
{
    return pre_calculated_corners_[static_cast<int>(rotation)];
}

void Building::pre_calculate_forms()
{
    for (int rotation = 0; rotation <= static_cast<int>(turnable_); ++rotation) {
        pre_calculated_forms_.push_back(rotate_form(default_form_, static_cast<Rotation>(rotation)));
    }
}

void Building::pre_calculate_corners()
{
    for (const auto& form : pre_calculated_forms_) {

        std::set<Square> corner_set;

        for (const auto& square : form) {
            corner_set.insert({square.x + 1, square.y});
            corner_set.insert({square.x - 1, square.y});
            corner_set.insert({square.x, square.y + 1});
            corner_set.insert({square.x, square.y - 1});
        }

        for (const auto& square : form) {
            corner_set.erase(square);
        }

        pre_calculated_corners_.push_back(std::vector<Square>(corner_set.begin(), corner_set.end()));
    }
}

const std::vector<Square> Building::translate_positions(const std::vector<Square> &positions, Square pos) const
{
    std::vector<Square> translated_positions;

    for (const auto& square : positions)
    {
        translated_positions.push_back(square + pos);
    }

    return translated_positions;
}


Move::Move(const Square &position, const BuildingType &type, Rotation rotation)
: pos(position), building_type(type), rotation(rotation) 
{

    const Building& building = Building::get_instance(type);

    int max_rotation = static_cast<int>(building.turnable());

    if (static_cast<int>(rotation) > max_rotation) {
        throw std::invalid_argument("Invalid rotation for the selected building.");
    }

    form = building.form(rotation, pos);
    corners = building.corners(rotation, pos);
}

int Move::encode() const
{
    return static_cast<int>(building_type) * board_size * max_rotations 
        + static_cast<int>(rotation) * board_size
        + pos.y * board_width + pos.x;
}

Move Move::decode_move(int action) {
    int building_type = action / (board_size * max_rotations);
    int rotation_idx = (action % (board_size * max_rotations)) / board_size;
    int y = (action % board_size) / board_width;
    int x = action % board_width;

    BuildingType type = static_cast<BuildingType>(building_type);
    Rotation rotation = static_cast<Rotation>(rotation_idx);
    Square position(x, y);

    return Move(position, type, rotation);
}

bool Move::operator==(const Move& other) const {
    return this->building_type == other.building_type &&
           this->rotation == other.rotation &&
           this->pos == other.pos;
}

const CellState get_square_color(const Move& move, Player player)
{
    if (move.building_type == BuildingType::Cathedral)
        return CellState::BLUE;
    else if (player == 0)
        return CellState::WHITE;
    else if (player == 1)
        return CellState::BLACK;

    return CellState::EMPTY; // Default case, though it should not occur
}

// Mapping of C++ client Building ids and player to Java client Building ids
const int player_building_to_java_building_id(BuildingType type, int player)
{
    switch (type) {

        case BuildingType::Tavern:          return player == 1 ? 1 : 12;
        case BuildingType::Stable:          return player == 1 ? 2 : 13;
        case BuildingType::Inn:             return player == 1 ? 3 : 14;
        case BuildingType::Bridge:          return player == 1 ? 4 : 15;
        case BuildingType::Manor:           return player == 1 ? 5 : 16;
        case BuildingType::Square:          return player == 1 ? 6 : 17;
        case BuildingType::Infirmary:       return player == 1 ? 8 : 19;
        case BuildingType::Castle:          return player == 1 ? 9 : 20;
        case BuildingType::Tower:           return player == 1 ? 10 : 21;
        case BuildingType::Black_Abbey:     return 7;
        case BuildingType::Black_Academy:   return 11;
        case BuildingType::White_Abbey:     return 18;
        case BuildingType::White_Academy:   return 22;
        case BuildingType::Cathedral:       return 23;

        default:
            throw std::runtime_error("Unhandled building type in parse_building_for_java_client");
    }
}

const std::tuple<BuildingType, int> java_building_to_cpp_building_player(int building) {

    // If building id is > 11 we have the white_player else black
    int player = building > 11 ? 0 : 1;

    // mapping of java building id to cpp tuple<buildingtype, player_idx>
    switch (building) {
        case 1: case 12:    return std::make_tuple(BuildingType::Tavern, player);
        case 2: case 13:    return std::make_tuple(BuildingType::Stable, player);
        case 3: case 14:    return std::make_tuple(BuildingType::Inn, player);
        case 4: case 15:    return std::make_tuple(BuildingType::Bridge, player);
        case 5: case 16:    return std::make_tuple(BuildingType::Manor, player);
        case 6: case 17:    return std::make_tuple(BuildingType::Square, player);
        case 7:             return std::make_tuple(BuildingType::Black_Abbey, player);
        case 18:            return std::make_tuple(BuildingType::White_Abbey, player);
        case 8: case 19:    return std::make_tuple(BuildingType::Infirmary, player);
        case 9: case 20:    return std::make_tuple(BuildingType::Castle, player);
        case 10: case 21:   return std::make_tuple(BuildingType::Tower, player);
        case 11:            return std::make_tuple(BuildingType::Black_Academy, player);
        case 22:            return std::make_tuple(BuildingType::White_Academy, player);
        case 23:            return std::make_tuple(BuildingType::Cathedral, player);
        default:            std::out_of_range("Invalid Building id. Building id range 1-23");
    }

    return {};
}

// [input, output] -> [0, ROTATE_0], [90, ROTATE_90], [180, ROTATE_180], [270, ROTATE_270]
const Rotation parse_rotation_angle(int angle)
{
    return static_cast<Rotation>(angle / 90);
}

const std::vector<Move> CathedralState::get_possible_moves() const
{
    return get_possible_moves(current_player_);
}

const std::vector<Move> CathedralState::get_possible_moves(Player player) const
{
    if (player != current_player_) {
        return {};
    }

    if (get_move_count() == 0)
        return get_possible_moves(white_specific_buildings[0], player);
    else
        return generate_all_possible_moves();
}

const std::vector<Move> CathedralState::get_possible_moves(const BuildingType &type, Player player) const
{
    if (!players[player].is_building_available(type))
        return {};
    
    std::vector<Move> possible_moves;

    const Building& building = Building::get_instance(type);

    int max_rotation = static_cast<int>(building.turnable());

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {

            for (int rotation_idx = 0; rotation_idx <= max_rotation; ++rotation_idx) {

                Rotation rotation = static_cast<Rotation>(rotation_idx);
                Move current_move({x, y}, type, rotation);

                if (board.is_move_valid(current_move, player)) {
                    possible_moves.push_back(current_move);
                }
            }
        }
    }

    return possible_moves;
}

void CathedralState::undo_move() {
  if (!history_.empty()) {

      history_.pop_back();
      --move_number_;
      
      board = initial_board;
      current_player_ = 0;
      removed_moves.clear();
      players[0].reset_building_availability();
      players[1].reset_building_availability();

      for (const auto& player_action : history_) {
          make_unvalidated_move(Move(player_action.action));
      }
  }
}

const bool CathedralState::make_move(const Move &move)
{
  if (!board.is_move_valid(move, current_player_))
      return false;

  make_unvalidated_move(move);

  return true;
}

void CathedralState::make_unvalidated_move(const Move &move)
{
  // update board state
  place_building(move);

  // reduce building availability
  players[current_player_].use_building(move.building_type);

  update_current_player();
}

const bool CathedralState::is_finished() const {
    //we check whether one player has still a move to be made left
    for (int i = 0; i < 2; ++i) {
        
        Player player_to_check = (current_player_ + i) % 2;

        for (const auto& type : players[player_to_check].get_available_building_types()) {
            if (!get_possible_moves(type, player_to_check).empty()) {
                return false;
            }
        }
    }

    return true;
}


void CathedralState::update_current_player()
{
    // Calculate the next player
    int next_player = (current_player_ + 1) % 2;

    // Check if the next player has available moves
    bool has_moves = false;
    for (const auto& type : players[next_player].get_available_building_types()) {
        if (!get_possible_moves(type, next_player).empty()) {
            has_moves = true;
            break;
        }
    }

    if (has_moves) {
        current_player_ = next_player;
    }
    // Otherwise, current_player_ remains the same, allowing them to continue playing
}


const std::vector<Move> CathedralState::generate_all_possible_moves() const
{
    std::vector<Move> all_possible_moves;

    const auto& available_buildings = players[current_player_].get_available_building_types();

    for (const auto& type : available_buildings) {
        const auto& moves = get_possible_moves(type, current_player_);
        all_possible_moves.insert(all_possible_moves.end(), moves.begin(), moves.end());
    }

    return all_possible_moves;
}

std::tuple<int, int> CathedralState::calc_score() const {
    const int initial_score = 47;
    int white_score = initial_score;
    int black_score = initial_score;

    for (const auto& pair : history_) {

        if (is_piece_removed(pair))
            continue;

        const auto& move = Move(pair.action);

        CellState player_color = get_square_color(move, pair.player);

        if (player_color == CellState::BLACK) {
            black_score -= move.form.size();
        } else if (player_color == CellState::WHITE) {
            white_score -= move.form.size();
        }
    }

    return std::make_tuple(white_score, black_score);
}

bool CathedralState::remove_move(const PlayerMove& player_move)
{
    for (const auto& square : player_move.move.form) {
        
        if (board.is_on_board(square)) {
            board.field[square.y][square.x] = CellState::EMPTY;
        }
    }

    removed_moves.push_back(player_move);

    players[player_move.player].return_building(player_move.move.building_type);

    return true;
}

bool CathedralState::place_building(const Move &move)
{
    board.place_color(move.form, get_square_color(move, current_player_));

    // Check for connections and potentially build regions
    if (get_move_count() > 2) {

        int number_of_connections = 0;

        for (const auto& square : move.corners) {

            if (!board.is_on_board(square) || board.field[square.y][square.x] == get_square_color(move, current_player_)|| get_move_count() == 3) {

                number_of_connections++;

                if (number_of_connections > 1) {
                    build_regions();
                    break;
                }
            }
        }
    }

    return true;
}

void CathedralState::process_region(const std::vector<Square>& region, CellState color) {

    const auto& player_moves_in_region = get_all_enemy_buildings_in_region(region, color);

    if (player_moves_in_region.size() < 2)
    {
        for (const auto& player_move : player_moves_in_region)
        {
            remove_move(player_move);
        }

        CellState owned_color = get_sub_color(color);

        for (const auto& square : region)
        {
            board.field[square.y][square.x] = owned_color;
        }
    }
}

const std::vector<PlayerMove> CathedralState::get_all_enemy_buildings_in_region(const std::vector<Square> &region, CellState enemy_color) const
{
    std::vector<PlayerMove> enemy_buildings;

    for (const auto& player_action : history_) {

        if (is_piece_removed(player_action))
            continue;

        const auto& move = Move(player_action.action);

        if (is_in_region(move.pos, region) && get_square_color(move, player_action.player) != enemy_color)
            enemy_buildings.push_back({player_action.player, move});
    }
        
    return enemy_buildings;
}

void CathedralState::build_regions()
{
    std::array<CellState, 2> colors = {CellState::BLACK, CellState::WHITE};

    for (auto color : colors) {

        std::array<std::array<int, 10>, 10> field_without_color{};
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                if (board.field[y][x] != color) {
                    field_without_color[y][x] = 1;
                }
            }
        }

        int runner = 0;
        while (runner < 100) {
            int x = runner % 10;
            int y = runner / 10;
            if (field_without_color[y][x] == 1) {
                std::queue<Square> free_fields_to_look_at;
                free_fields_to_look_at.push(Square(x, y));
                field_without_color[y][x] = 0;

                std::vector<Square> region;
                while (!free_fields_to_look_at.empty()) {
                    Square current_field = free_fields_to_look_at.front();
                    free_fields_to_look_at.pop();

                    region.push_back(current_field);

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx != 0 || dy != 0) {
                                const Square square = {current_field.x + dx, current_field.y + dy};
                                if (board.is_on_board(square) && field_without_color[square.y][square.x] == 1) {
                                    free_fields_to_look_at.push(Square(square.x, square.y));
                                    field_without_color[square.y][square.x] = 0;
                                }
                            }
                        }
                    }
                }

                process_region(region, color);
            }

            runner++;
        }
    }
}

bool CathedralState::is_in_region(const Square &position, const std::vector<Square> &region) const
{
    return std::find(region.begin(), region.end(), position) != region.end();
}

bool operator==(const PlayerMove &lhs, const PlayerMove &rhs)
{
    return lhs.player == rhs.player && lhs.move == rhs.move;
}

CellState CathedralState::get_sub_color(CellState color) const
{
    switch (color) {
        case CellState::BLACK:
            return CellState::BLACK_REGION;
        case CellState::WHITE:
            return CellState::WHITE_REGION;
        default:
            return CellState::EMPTY;
    }
}

void CathedralState::PopulateFreeSquaresPlane(TensorView<3>& view, Player player) const {
    CellState player_piece_color = (player == 0) ? CellState::WHITE : CellState::BLACK;
    CellState player_region_color = get_sub_color(player_piece_color);

    for (int y = 0; y < board_width; ++y) {
        for (int x = 0; x < board_width; ++x) {

            CellState cell_state = board.field[y][x];

            if (board.color_is_compatible(cell_state, player_piece_color) ||
                board.color_is_compatible(cell_state, player_region_color)) {
                view[{15, y, x}] = 1.0f;
            } else {
                view[{15, y, x}] = 0.0f;
            }
        }
    }
}

void CathedralState::PopulateCurrentPlayerPlane(TensorView<3> &view, Player player) const
{
    float value = (player == 1) ? 1.0f : 0.0f;

    for (int y = 0; y < board_width; ++y) {
        for (int x = 0; x < board_width; ++x) {
            view[{15, y, x}] = value;
        }
    }
}

}  // namespace cathedral
}  // namespace open_spiel
