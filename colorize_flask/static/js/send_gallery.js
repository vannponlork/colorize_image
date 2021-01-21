<script>
            $(function() {
                          $('#g1').click(function() {
                          var form_data = new FormData($('#upload-file')[0]);
                          $.ajax({
                              type: 'POST',
                              url: '/uploadajax',
                              data: form_data,
                              contentType: false,
                              cache: false,
                              processData: false,
                              success: function(data) {
                                  console.log('Success!');
                                },
                            });
                        });
                    });
    </script>