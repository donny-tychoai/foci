import json
import yaml

def write_config(foci_json_path:str, output_path:str):

    with open(foci_json_path, "r") as f:
        json_obj = json.load(f)

    yaml_dump = {
        "num_waypoints" : len(json_obj) // 9
    }
    
    curr_waypoint = 0
    for i, entry in enumerate(json_obj):
        if i % 9 == 0:
            yaml_dump[f"waypoint{curr_waypoint}"] = {"x": entry[0], "y": entry[1], "z": entry[2] - 2}
            curr_waypoint += 1

    with open(output_path, 'w') as file:
        yaml.dump(yaml_dump, file, default_flow_style=False)
        print(f"Successfully wrote data to {output_path}")


if __name__ == "__main__":
    write_config("submodules/foci/planned_path.json", "obstacle_path_1.yaml")