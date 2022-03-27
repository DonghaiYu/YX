
/**
 * load new function in menu
 * @param {string} name function name 
 */
function load_new_func(name) {
    $("#operating_place").load(`static/operating_pages/${name}.html`, function () {
        console.log(`load ${name} finish`);
    });
    $("#show_place").html("");
}

/**
 * load function explaination
 * @param {string} name function name
 */
function about(name) {
    $("#operating_place").load('static/operating_pages/about_' + name + '.html', function () {
        console.log("load fisher test");
    });
}

/**
 * submit model request
 * @param {string} model_class model class
 */
function request_model(model_class) {
    var param_data = new FormData($("#model_params")[0]);
    var model_url = "/model"

    param_data.append("model_class", model_class);

    $.ajax({
        type: "POST",
        url: model_url,
        data: param_data,
        async: false,
        processData: false,
        contentType: false,
        error: function(request) {
            alert("error");
        },
        success: function(result_data){
            $("#show_place").load('static/result_page.html', function () {
                console.log("load result finish");
                const vue_result = {
                    data() {
                      result_data["img_data"] = encodeURI(result_data["img_data"])
                      return result_data;
                    }
                }
                  
                Vue.createApp(vue_result).mount('#result_place');
            });
            
        }
    })
}